import { describe, expect, it } from "vitest";
import { CodexAgentRuntime, parseCommandLine } from "./codex-runtime.js";

const fakeAppServer = String.raw`
import { createInterface } from "node:readline";

const input = createInterface({ input: process.stdin });
let dynamicTool = "rakazo_echo";

function send(message) {
  process.stdout.write(JSON.stringify(message) + "\n");
}

input.on("line", (line) => {
  const message = JSON.parse(line);
  if (message.method === "initialize") {
    send({ id: message.id, result: { userAgent: "fake-codex", codexHome: "/tmp", platformFamily: "unix", platformOs: "test" } });
    return;
  }
  if (message.method === "thread/start") {
    dynamicTool = message.params.dynamicTools[0].name;
    send({ id: message.id, result: { thread: { id: "thread-1" } } });
    return;
  }
  if (message.method === "turn/start") {
    send({ id: message.id, result: { turn: { id: "turn-1", status: "inProgress" } } });
    send({ method: "turn/started", params: { threadId: "thread-1", turn: { id: "turn-1" } } });
    send({ id: 90, method: "item/tool/call", params: { callId: "call-1", threadId: "thread-1", turnId: "turn-1", tool: dynamicTool, arguments: { text: "hello" } } });
    return;
  }
  if (message.id === 90) {
    if (!message.result.success) throw new Error("dynamic tool failed");
    send({ method: "item/agentMessage/delta", params: { threadId: "thread-1", turnId: "turn-1", itemId: "item-1", delta: "codex-ok" } });
    send({ method: "thread/tokenUsage/updated", params: { threadId: "thread-1", turnId: "turn-1", tokenUsage: { last: { inputTokens: 7, outputTokens: 3 }, total: { inputTokens: 7, outputTokens: 3 } } } });
    send({ method: "turn/completed", params: { threadId: "thread-1", turn: { id: "turn-1", status: "completed" } } });
    setTimeout(() => process.exit(0), 5);
  }
});
`;

describe("Codex App Server runtime", () => {
  it("speaks JSONL, maps dynamic tools, and streams a completed turn", async () => {
    const runtime = new CodexAgentRuntime({
      command: process.execPath,
      args: ["--input-type=module", "-e", fakeAppServer],
    });
    const events = [] as Array<{ type: string; [key: string]: unknown }>;
    const calls: Array<{ name: string; args: Record<string, unknown>; executionId: string }> = [];

    for await (const event of runtime.run(
      {
        botId: "bot",
        threadId: "thread",
        runId: "run",
        workspaceRoot: process.cwd(),
        prompt: "Say hello",
        instructions: "Be concise.",
        history: [],
        tools: [
          {
            name: "echo",
            description: "Echo text",
            inputSchema: {
              type: "object",
              properties: { text: { type: "string" } },
              required: ["text"],
            },
          },
        ],
        model: { provider: "openrouter", id: "test-model" },
        executeTool: async (name, args, executionId) => {
          calls.push({ name, args, executionId });
          return { echoed: args.text };
        },
      },
      {
        operationId: "run",
        traceId: "run",
        workspaceId: "workspace",
        userId: "user",
        signal: new AbortController().signal,
      },
    )) {
      events.push(event);
    }

    expect(calls).toEqual([{ name: "echo", args: { text: "hello" }, executionId: "call-1" }]);
    expect(events.some((event) => event.type === "tool" && event.name === "echo")).toBe(true);
    expect(events.some((event) => event.type === "text" && event.text === "codex-ok")).toBe(true);
    expect(
      events.some(
        (event) => event.type === "usage" && event.inputTokens === 7 && event.outputTokens === 3,
      ),
    ).toBe(true);
    expect(events.at(-1)).toEqual({ type: "done", text: "codex-ok" });
  });

  it("parses a configurable app-server command without invoking a shell", () => {
    expect(parseCommandLine('app-server --listen "ws://127.0.0.1:4500"')).toEqual([
      "app-server",
      "--listen",
      "ws://127.0.0.1:4500",
    ]);
    expect(() => parseCommandLine("app-server 'unterminated")).toThrow(/unterminated/i);
  });

  it("reports a synchronous process startup failure as runtime events", async () => {
    const runtime = new CodexAgentRuntime({
      spawn: (() => {
        throw new Error("codex executable is unavailable");
      }) as never,
    });
    const events = [] as Array<{ type: string; [key: string]: unknown }>;

    for await (const event of runtime.run(
      {
        botId: "bot",
        threadId: "thread",
        runId: "run-startup-error",
        prompt: "Say hello",
        instructions: "Be concise.",
        history: [],
        tools: [],
        model: { provider: "codex", id: "gpt-5" },
      },
      {
        operationId: "run",
        traceId: "run",
        workspaceId: "workspace",
        userId: "user",
        signal: new AbortController().signal,
      },
    )) {
      events.push(event);
    }

    expect(events).toEqual([
      { type: "text", text: "I hit a problem: codex executable is unavailable" },
      { type: "done", text: "codex executable is unavailable" },
    ]);
  });
});
