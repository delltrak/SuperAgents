import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import path from "node:path";
import { createInterface } from "node:readline";
import type {
  AdapterContext,
  AgentRunRequest,
  AgentRuntime,
  AgentRuntimeEvent,
  ConnectorTool,
} from "@rakazo/adapter-kit";

type JsonRpcId = string | number;

type AppServerMessage = {
  id?: JsonRpcId;
  method?: string;
  params?: unknown;
  result?: unknown;
  error?: { code?: number; message?: string; data?: unknown };
};

type ToolBinding = {
  codexName: string;
  rakazoName: string;
};

type ActiveCodexRun = {
  client: CodexAppServerClient;
  threadId?: string;
  turnId?: string;
  abortRequested: boolean;
};

export interface CodexAgentRuntimeOptions {
  command?: string;
  args?: string[];
  env?: NodeJS.ProcessEnv;
  spawn?: typeof spawn;
}

/**
 * Runs Rakazo turns through the Codex App Server JSON-RPC protocol.
 *
 * Rakazo's own dynamic tools remain the authority for writes, shell commands,
 * memory, takeover, and connected apps. Native Codex execution is deliberately
 * read-only so this adapter does not bypass the configured Rakazo sandbox.
 */
export class CodexAgentRuntime implements AgentRuntime {
  private readonly command: string;
  private readonly args: string[];
  private readonly env: NodeJS.ProcessEnv;
  private readonly spawnProcess: typeof spawn;
  private readonly running = new Map<string, ActiveCodexRun>();

  constructor(options: CodexAgentRuntimeOptions = {}) {
    this.command = options.command ?? process.env.CODEX_APP_SERVER_BIN ?? "codex";
    this.args =
      options.args ?? parseCommandLine(process.env.CODEX_APP_SERVER_ARGS ?? "app-server --stdio");
    this.env = { ...process.env, ...options.env };
    this.spawnProcess = options.spawn ?? spawn;
  }

  describe() {
    return {
      id: "codex-app-server",
      contractVersion: "1",
      adapterVersion: "0.1.0",
      capabilities: { streaming: true, compaction: true, tools: true, scripted: false },
    };
  }

  async abort(runId: string): Promise<void> {
    const active = this.running.get(runId);
    if (!active) return;
    active.abortRequested = true;

    if (active.threadId && active.turnId) {
      await withTimeout(
        active.client.request("turn/interrupt", {
          threadId: active.threadId,
          turnId: active.turnId,
        }),
        750,
      ).catch(() => undefined);
    }
    await active.client.close();
  }

  async *run(request: AgentRunRequest, context: AdapterContext): AsyncIterable<AgentRuntimeEvent> {
    if (context.signal.aborted) {
      yield { type: "done", text: "stopped" };
      return;
    }
    let child: ChildProcessWithoutNullStreams;
    try {
      child = this.spawnProcess(this.command, this.args, {
        stdio: ["pipe", "pipe", "pipe"],
        env: this.env,
      });
    } catch (error) {
      const message = sanitizeError(error instanceof Error ? error.message : String(error));
      yield { type: "text", text: `I hit a problem: ${message}` };
      yield { type: "done", text: message };
      return;
    }
    const client = new CodexAppServerClient(child);
    const active: ActiveCodexRun = { client, abortRequested: false };
    this.running.set(request.runId, active);

    const onAbort = () => {
      void this.abort(request.runId);
    };
    context.signal.addEventListener("abort", onAbort, { once: true });

    try {
      const bindings = createToolBindings(request.tools);
      await client.request("initialize", {
        clientInfo: {
          name: "rakazo",
          title: "Rakazo",
          version: "0.1.0",
        },
        capabilities: {
          experimentalApi: true,
        },
      });
      client.notify("initialized", {});

      const thread = await client.request<Record<string, unknown>>("thread/start", {
        ...threadStartParams(request, bindings, this.env.CODEX_DEFAULT_MODEL),
        ...(request.workspaceRoot
          ? {
              cwd: path.resolve(request.workspaceRoot),
              runtimeWorkspaceRoots: [path.resolve(request.workspaceRoot)],
            }
          : {}),
      });
      const threadResponse = readRecord(thread);
      active.threadId = readString(threadResponse.thread, "id");
      if (!active.threadId) throw new Error("Codex App Server did not return a thread id");
      const codexProvider = readString(threadResponse, "modelProvider") ?? request.model.provider;
      const codexModelId = readString(threadResponse, "model") ?? request.model.id;

      const turn = await client.request<Record<string, unknown>>("turn/start", {
        threadId: active.threadId,
        input: [{ type: "text", text: formatPrompt(request) }],
      });
      active.turnId = readString(readRecord(turn).turn, "id");

      yield { type: "progress", text: "working…" };

      let streamed = "";
      let completed = false;
      let latestUsage: { inputTokens: number; outputTokens: number } | undefined;

      for await (const message of client.notifications) {
        if (!message.method) continue;

        if (message.method === "item/tool/call") {
          const params = readRecord(message.params);
          const codexName = readString(params, "tool") ?? "";
          const binding = bindings.byCodexName.get(codexName);
          const args = readRecord(params.arguments);
          const executionId = readString(params, "callId") ?? `${request.runId}:${codexName}`;

          if (!binding) {
            await client.respondError(
              message.id,
              -32602,
              `Unknown Rakazo dynamic tool: ${codexName}`,
            );
            continue;
          }

          yield {
            type: "tool",
            name: binding.rakazoName,
            args,
            executionId,
          };

          let result: unknown;
          try {
            result = request.executeTool
              ? await request.executeTool(binding.rakazoName, args, executionId)
              : { error: "Rakazo tool executor is unavailable" };
          } catch (error) {
            result = {
              error: sanitizeError(error instanceof Error ? error.message : String(error)),
            };
          }

          const failed = isToolError(result);
          await client.respond(message.id, {
            success: !failed,
            contentItems: [
              {
                type: "inputText",
                text: summarizeToolResult(result),
              },
            ],
          });

          if (binding.rakazoName === "request_takeover") {
            yield {
              type: "takeover",
              reason: String(args.reason ?? "I need you on the screen."),
            };
            return;
          }
          continue;
        }

        const serverRequest = await this.handleServerRequest(client, message);
        if (serverRequest.ask) {
          yield serverRequest.ask;
          return;
        }
        if (serverRequest.progress) yield { type: "progress", text: serverRequest.progress };

        if (message.method === "item/agentMessage/delta") {
          const delta = readString(readRecord(message.params), "delta");
          if (delta) {
            streamed += delta;
            yield { type: "text", text: delta };
          }
          continue;
        }

        if (message.method === "item/completed") {
          const item = readRecord(readRecord(message.params).item);
          if (item.type === "agentMessage" && typeof item.text === "string" && !streamed) {
            streamed = item.text;
            yield { type: "text", text: item.text };
          }
          continue;
        }

        if (message.method === "thread/tokenUsage/updated") {
          const tokenUsage = readRecord(readRecord(message.params).tokenUsage);
          const last = readRecord(tokenUsage.last);
          const inputTokens = readNumber(last, "inputTokens");
          const outputTokens = readNumber(last, "outputTokens");
          if (inputTokens !== undefined && outputTokens !== undefined) {
            latestUsage = { inputTokens, outputTokens };
          }
          continue;
        }

        if (message.method === "turn/started") {
          active.turnId =
            readString(readRecord(readRecord(message.params).turn), "id") ?? active.turnId;
          continue;
        }

        if (message.method === "turn/completed") {
          const turnParams = readRecord(message.params);
          const turnState = readRecord(turnParams.turn);
          const status = String(turnState.status ?? "completed");
          if (status !== "completed") {
            if (status === "interrupted" && active.abortRequested) return;
            throw new Error(
              String(
                readString(readRecord(turnState.error), "message") ??
                  `Codex turn ended with status ${status}`,
              ),
            );
          }
          completed = true;
          if (latestUsage) {
            yield {
              type: "usage",
              ...latestUsage,
              provider: codexProvider,
              model: codexModelId,
            };
          }
          yield { type: "done", ...(streamed ? { text: streamed } : {}) };
          break;
        }

        if (message.method === "error") {
          const params = readRecord(message.params);
          if (params.willRetry === true) {
            yield { type: "progress", text: "Codex is retrying the turn…" };
            continue;
          }
          throw new Error(
            readString(readRecord(params.error), "message") ??
              readString(params, "message") ??
              "Codex App Server returned an error",
          );
        }
      }

      if (!completed && !active.abortRequested && !context.signal.aborted) {
        throw new Error(`Codex App Server exited before completing run${client.stderrSuffix()}`);
      }
    } catch (error) {
      if (!active.abortRequested && !context.signal.aborted) {
        const message = sanitizeError(error instanceof Error ? error.message : String(error));
        yield { type: "text", text: `I hit a problem: ${message}` };
        yield { type: "done", text: message };
      }
    } finally {
      context.signal.removeEventListener("abort", onAbort);
      this.running.delete(request.runId);
      await client.close();
    }
  }

  private async handleServerRequest(
    client: CodexAppServerClient,
    message: AppServerMessage,
  ): Promise<{ ask?: Extract<AgentRuntimeEvent, { type: "ask" }>; progress?: string }> {
    switch (message.method) {
      case "item/tool/requestUserInput": {
        const params = readRecord(message.params);
        const questions = Array.isArray(params.questions) ? params.questions : [];
        const text = questions
          .map((question) => readString(readRecord(question), "question"))
          .filter((question): question is string => Boolean(question))
          .join("\n");
        return {
          ask: {
            type: "ask",
            text: text || "Codex needs more information to continue.",
            detail: "Answer in the Rakazo thread and retry the run.",
          },
        };
      }
      case "item/commandExecution/requestApproval":
      case "item/fileChange/requestApproval":
      case "execCommandApproval":
      case "applyPatchApproval":
        await client.respond(message.id, { decision: "decline" });
        return { progress: "Codex requested a native host action; Rakazo declined it." };
      case "item/permissions/requestApproval":
        await client.respond(message.id, {
          permissions: { fileSystem: null, network: null },
          scope: "turn",
        });
        return { progress: "Codex requested additional permissions; Rakazo granted none." };
      case "mcpServer/elicitation/request":
        await client.respond(message.id, { action: "decline", content: null });
        return { progress: "Codex requested MCP input; Rakazo declined it." };
      case "currentTime/read":
        await client.respond(message.id, { currentTimeAt: Math.floor(Date.now() / 1000) });
        return {};
      default:
        if (message.id !== undefined) {
          await client.respondError(
            message.id,
            -32601,
            `Rakazo does not handle Codex server request ${message.method ?? "unknown"}`,
          );
        }
        return {};
    }
  }
}

class CodexAppServerClient {
  readonly notifications: AsyncIterable<AppServerMessage>;
  private readonly queue = new AsyncQueue<AppServerMessage>();
  private readonly pending = new Map<
    JsonRpcId,
    { resolve: (result: unknown) => void; reject: (error: Error) => void }
  >();
  private readonly lines;
  private nextId = 1;
  private closed = false;
  private stderr = "";

  constructor(private readonly child: ChildProcessWithoutNullStreams) {
    this.notifications = this.queue;
    this.lines = createInterface({ input: child.stdout });
    this.lines.on("line", (line) => this.handleLine(line));
    child.stderr.on("data", (chunk: Buffer | string) => {
      this.stderr = `${this.stderr}${String(chunk)}`.slice(-4000);
    });
    child.on("error", (error) => this.fail(error));
    child.on("exit", (code, signal) => {
      if (code !== 0 || signal) {
        this.fail(new Error(`Codex App Server exited (${code ?? signal ?? "unknown"})`));
      } else {
        this.finish();
      }
    });
  }

  request<T>(method: string, params: unknown): Promise<T> {
    if (this.closed) return Promise.reject(new Error("Codex App Server connection is closed"));
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (result: unknown) => void, reject });
      try {
        this.write({ id, method, params });
      } catch (error) {
        this.pending.delete(id);
        reject(error instanceof Error ? error : new Error(String(error)));
      }
    });
  }

  notify(method: string, params: unknown): void {
    this.write({ method, params });
  }

  respond(id: JsonRpcId | undefined, result: unknown): Promise<void> {
    if (id === undefined) return Promise.resolve();
    this.write({ id, result });
    return Promise.resolve();
  }

  respondError(id: JsonRpcId | undefined, code: number, message: string): Promise<void> {
    if (id === undefined) return Promise.resolve();
    this.write({ id, error: { code, message } });
    return Promise.resolve();
  }

  stderrSuffix(): string {
    const suffix = this.stderr.trim();
    return suffix ? `: ${sanitizeError(suffix)}` : "";
  }

  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    this.lines.close();
    const error = new Error("Codex App Server connection closed");
    for (const waiter of this.pending.values()) waiter.reject(error);
    this.pending.clear();
    this.queue.close();
    if (!this.child.killed) this.child.kill("SIGTERM");
  }

  private handleLine(line: string): void {
    if (this.closed || !line.trim()) return;
    let message: AppServerMessage;
    try {
      message = JSON.parse(line) as AppServerMessage;
    } catch {
      this.fail(new Error(`Codex App Server emitted invalid JSON: ${line.slice(0, 200)}`));
      return;
    }

    if (message.method) {
      this.queue.push(message);
      return;
    }

    if (message.id !== undefined && this.pending.has(message.id)) {
      const waiter = this.pending.get(message.id);
      this.pending.delete(message.id);
      if (!waiter) return;
      if (message.error) {
        waiter.reject(
          new Error(
            `Codex App Server error ${message.error.code ?? "unknown"}: ${message.error.message ?? "request failed"}`,
          ),
        );
      } else {
        waiter.resolve(message.result);
      }
      return;
    }
  }

  private write(message: AppServerMessage): void {
    if (this.closed) throw new Error("Codex App Server connection is closed");
    this.child.stdin.write(`${JSON.stringify(message)}\n`);
  }

  private fail(error: Error): void {
    if (this.closed) return;
    for (const waiter of this.pending.values()) waiter.reject(error);
    this.pending.clear();
    this.queue.close();
  }

  private finish(): void {
    if (this.closed) return;
    this.closed = true;
    const error = new Error("Codex App Server closed the connection");
    for (const waiter of this.pending.values()) waiter.reject(error);
    this.pending.clear();
    this.queue.close();
  }
}

class AsyncQueue<T> implements AsyncIterable<T> {
  private readonly items: T[] = [];
  private waiter: (() => void) | undefined;
  private closed = false;

  push(item: T): void {
    if (this.closed) return;
    this.items.push(item);
    this.waiter?.();
  }

  close(): void {
    this.closed = true;
    this.waiter?.();
  }

  async *[Symbol.asyncIterator](): AsyncIterator<T> {
    while (!this.closed || this.items.length) {
      if (this.items.length) {
        yield this.items.shift() as T;
        continue;
      }
      await new Promise<void>((resolve) => {
        this.waiter = resolve;
      });
      this.waiter = undefined;
    }
  }
}

function createToolBindings(tools: ConnectorTool[]) {
  const used = new Set<string>();
  const byCodexName = new Map<string, ToolBinding>();
  const dynamicTools = tools.map((tool, index) => {
    const codexName = uniqueCodexToolName(tool.name, index, used);
    const binding = { codexName, rakazoName: tool.name };
    byCodexName.set(codexName, binding);
    return {
      type: "function",
      name: codexName,
      description: tool.description,
      inputSchema: tool.inputSchema,
    };
  });
  return { dynamicTools, byCodexName };
}

function uniqueCodexToolName(name: string, index: number, used: Set<string>): string {
  const normalized =
    name.replace(/[^a-zA-Z0-9_-]/g, "_").replace(/^_+|_+$/g, "") || `tool_${index}`;
  const base = `rakazo_${normalized}`.slice(0, 64);
  let candidate = base;
  let suffix = 2;
  while (used.has(candidate)) {
    const tail = `_${suffix}`;
    candidate = `${base.slice(0, 64 - tail.length)}${tail}`;
    suffix += 1;
  }
  used.add(candidate);
  return candidate;
}

function threadStartParams(
  request: AgentRunRequest,
  bindings: { dynamicTools: Array<Record<string, unknown>> },
  configuredModel?: string,
): Record<string, unknown> {
  const model = codexModel(request, configuredModel);
  return {
    ...(model ? { model } : {}),
    baseInstructions: [
      request.instructions,
      "Use Rakazo dynamic tools for all work that changes files, runs commands, stores memory, uses connected apps, or needs user takeover. Native Codex host actions are read-only in this integration.",
      bindings.dynamicTools.length
        ? `Rakazo dynamic tools are exposed with the names ${bindings.dynamicTools.map((tool) => String(tool.name)).join(", ")}.`
        : "No Rakazo dynamic tools are available.",
    ].join("\n\n"),
    approvalPolicy: "never",
    sandbox: "read-only",
    ephemeral: true,
    dynamicTools: bindings.dynamicTools,
  };
}

function codexModel(request: AgentRunRequest, configuredValue?: string): string | undefined {
  const configured = configuredValue?.trim();
  if (configured) return configured;
  if (["codex", "openai"].includes(request.model.provider) && request.model.id !== "scripted") {
    return request.model.id;
  }
  return undefined;
}

function formatPrompt(request: AgentRunRequest): string {
  const last = request.history.at(-1);
  const history =
    last?.role === "user" && last.content === request.prompt
      ? request.history.slice(0, -1)
      : request.history;
  const prior = history
    .filter((message) => message.content.trim())
    .map((message) => `${message.role.toUpperCase()}: ${message.content}`)
    .join("\n\n");
  return [
    prior ? `Prior conversation from Rakazo (context only):\n\n${prior}` : "",
    `CURRENT USER REQUEST:\n\n${request.prompt}`,
  ]
    .filter(Boolean)
    .join("\n\n");
}

function readRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function readString(value: unknown, key: string): string | undefined {
  const candidate = readRecord(value)[key];
  return typeof candidate === "string" && candidate ? candidate : undefined;
}

function readNumber(value: unknown, key: string): number | undefined {
  const candidate = readRecord(value)[key];
  return typeof candidate === "number" && Number.isFinite(candidate) ? candidate : undefined;
}

function isToolError(value: unknown): boolean {
  return Boolean(value && typeof value === "object" && "error" in value);
}

function summarizeToolResult(result: unknown): string {
  try {
    const text = JSON.stringify(result);
    if (!text) return "ok";
    return text.length > 12_000 ? `${text.slice(0, 12_000)}…` : text;
  } catch {
    return "ok";
  }
}

function sanitizeError(message: string): string {
  return message
    .replace(/sk-or-v1-[a-zA-Z0-9]+/g, "[redacted]")
    .replace(/sk-[a-zA-Z0-9-]+/g, "[redacted]")
    .replace(/Bearer\s+\S+/gi, "Bearer [redacted]")
    .replace(/COMPOSIO_API_KEY[=:]?\s*\S+/gi, "COMPOSIO_API_KEY=[redacted]");
}

export function parseCommandLine(input: string): string[] {
  const tokens: string[] = [];
  let token = "";
  let quote: "'" | '"' | undefined;
  let escaped = false;

  for (const character of input.trim()) {
    if (escaped) {
      token += character;
      escaped = false;
      continue;
    }
    if (character === "\\" && quote !== "'") {
      escaped = true;
      continue;
    }
    if (quote) {
      if (character === quote) quote = undefined;
      else token += character;
      continue;
    }
    if (character === "'" || character === '"') {
      quote = character;
      continue;
    }
    if (/\s/.test(character)) {
      if (token) {
        tokens.push(token);
        token = "";
      }
      continue;
    }
    token += character;
  }

  if (escaped) token += "\\";
  if (quote) throw new Error("Unterminated quote in CODEX_APP_SERVER_ARGS");
  if (token) tokens.push(token);
  return tokens;
}

async function withTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<T>((_, reject) => {
        timer = setTimeout(() => reject(new Error("Timed out")), timeoutMs);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}
