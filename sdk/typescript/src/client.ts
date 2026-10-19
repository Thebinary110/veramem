import type {
  AddOptions,
  AddResult,
  Answer,
  AnswerOptions,
  ApiKeyInfo,
  CheckActionResult,
  CreatedKey,
  CreateKeyOptions,
  DeleteOptions,
  ExportResult,
  FeedbackOptions,
  FeedbackResult,
  ForgetResult,
  GrantKind,
  GrantOptions,
  GrantResult,
  HealthResult,
  HistoryEntry,
  InfoResult,
  JsonObject,
  LearnOptions,
  LearnResult,
  ListOptions,
  MeResult,
  MemoryItem,
  Message,
  ResolvedEvidence,
  RetractResult,
  RevokeKeyResult,
  Scope,
  SearchOptions,
  SearchResult,
  ShareableResult,
  ShareOptions,
  TeamsResult,
  VerifyResult,
} from "./types.js";

export const DEFAULT_BASE_URL = "http://127.0.0.1:8765";
export const DEFAULT_TIMEOUT_MS = 60_000;

export interface VeraMemOptions {
  /** Server root, e.g. "http://127.0.0.1:8765". */
  baseUrl?: string;
  /** Sent as `Authorization: Bearer <apiKey>`. */
  apiKey?: string;
  /** Custom fetch implementation (defaults to the global fetch). */
  fetch?: typeof fetch;
  /** Per-request timeout in milliseconds (default 60000). 0 disables it. */
  timeoutMs?: number;
}

export interface RequestOptions {
  /** Abort this request early. Combined with the client timeout. */
  signal?: AbortSignal;
  /** Override the client timeout for this request. */
  timeoutMs?: number;
}

/** Thrown for non-2xx responses, timeouts and network failures. */
export class VeraMemError extends Error {
  /** HTTP status; 0 for network errors and timeouts. */
  readonly status: number;
  /** The server's `detail` field (or the raw body / underlying error message). */
  readonly detail: unknown;
  readonly method: string;
  readonly url: string;

  constructor(message: string, init: { status: number; detail: unknown; method: string; url: string; cause?: unknown }) {
    super(message);
    this.name = "VeraMemError";
    this.status = init.status;
    this.detail = init.detail;
    this.method = init.method;
    this.url = init.url;
    if (init.cause !== undefined) (this as { cause?: unknown }).cause = init.cause;
    Object.setPrototypeOf(this, new.target.prototype);
  }

  /** True when the request was aborted by the client timeout. */
  get isTimeout(): boolean {
    return this.status === 0 && this.detail === "timeout";
  }
}

type Query = Record<string, string | number | boolean | null | undefined>;

/** Drop undefined fields so the server applies its own defaults. */
function compact<T extends Record<string, unknown>>(obj: T): Partial<T> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(obj)) if (v !== undefined) out[k] = v;
  return out as Partial<T>;
}

function scopeQuery(scope?: Scope): Query {
  return { user_id: scope?.userId, agent_id: scope?.agentId };
}

/**
 * Client for the VeraMem REST API.
 *
 * ```ts
 * const mem = new VeraMem({ baseUrl: "http://127.0.0.1:8765", apiKey: process.env.VERAMEM_API_KEY });
 * await mem.add("I moved to Berlin last month", { userId: "alice" });
 * const r = await mem.search("where does the user live?", { userId: "alice" });
 * ```
 */
export class VeraMem {
  readonly baseUrl: string;
  readonly timeoutMs: number;
  private readonly apiKey: string | undefined;
  private readonly fetchImpl: typeof fetch;

  constructor(options: VeraMemOptions = {}) {
    this.baseUrl = (options.baseUrl ?? DEFAULT_BASE_URL).replace(/\/+$/, "");
    this.apiKey = options.apiKey;
    this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    const f = options.fetch ?? (typeof fetch === "function" ? fetch : undefined);
    if (!f) throw new Error("VeraMem: no global fetch available; pass options.fetch");
    // Bind so browsers don't throw "Illegal invocation" when called as a method.
    this.fetchImpl = options.fetch ? f : f.bind(globalThis);
  }

  // ------------------------------------------------------------- service
  health(opts?: RequestOptions): Promise<HealthResult> {
    return this.request("GET", "/health", { ...opts });
  }

  /** Server configuration; global counts are included only for admin callers. */
  info(opts?: RequestOptions): Promise<InfoResult> {
    return this.request("GET", "/v1/info", { ...opts });
  }

  /** Who the server thinks you are (role and pinned user/agent of your key). */
  me(opts?: RequestOptions): Promise<MeResult> {
    return this.request("GET", "/v1/me", { ...opts });
  }

  // --------------------------------------------------------------- write
  /** Ingest a string, a message, or a list of messages. */
  add(messages: string | Message | Message[], options: AddOptions = {}, opts?: RequestOptions): Promise<AddResult> {
    return this.request("POST", "/v1/memories", {
      ...opts,
      body: compact({
        messages,
        user_id: options.userId,
        agent_id: options.agentId,
        session_id: options.sessionId,
        team_id: options.teamId,
        infer: options.infer,
        timestamp: options.timestamp,
        metadata: options.metadata,
        resources: options.resources,
      }),
    });
  }

  /** Manually correct a memory; creates a new version and keeps history. */
  update(memoryId: string, text: string, scope: Scope = {}, opts?: RequestOptions): Promise<MemoryItem> {
    return this.request("PUT", `/v1/memories/${enc(memoryId)}`, {
      ...opts,
      body: compact({ text, user_id: scope.userId, agent_id: scope.agentId }),
    });
  }

  /** Owner consent for de-identified transfer of one memory. */
  setShareable(memoryId: string, shareable = true, options: { userId?: string } = {}, opts?: RequestOptions): Promise<ShareableResult> {
    return this.request("PUT", `/v1/memories/${enc(memoryId)}/shareable`, {
      ...opts,
      body: compact({ shareable, user_id: options.userId }),
    });
  }

  /** Retract a memory (kept for audit), or erase it with `{ hard: true }`. */
  delete(memoryId: string, options?: DeleteOptions & { hard?: false }, opts?: RequestOptions): Promise<RetractResult>;
  delete(memoryId: string, options: DeleteOptions & { hard: true }, opts?: RequestOptions): Promise<ForgetResult>;
  delete(memoryId: string, options?: DeleteOptions, opts?: RequestOptions): Promise<RetractResult | ForgetResult>;
  delete(memoryId: string, options: DeleteOptions = {}, opts?: RequestOptions): Promise<RetractResult | ForgetResult> {
    return this.request("DELETE", `/v1/memories/${enc(memoryId)}`, {
      ...opts,
      query: { ...scopeQuery(options), hard: options.hard ? true : undefined },
    });
  }

  /** Permanently erase one memory (hard delete). */
  forget(memoryId: string, scope: Scope = {}, opts?: RequestOptions): Promise<ForgetResult> {
    return this.delete(memoryId, { ...scope, hard: true }, opts);
  }

  /** Erase everything stored about a user. */
  forgetUser(userId: string, opts?: RequestOptions): Promise<ForgetResult> {
    return this.request("DELETE", `/v1/users/${enc(userId)}`, { ...opts });
  }

  // ---------------------------------------------------------------- read
  get(memoryId: string, scope: Scope = {}, opts?: RequestOptions): Promise<MemoryItem> {
    return this.request("GET", `/v1/memories/${enc(memoryId)}`, { ...opts, query: scopeQuery(scope) });
  }

  list(options: ListOptions = {}, opts?: RequestOptions): Promise<MemoryItem[]> {
    return this.request("GET", "/v1/memories", {
      ...opts,
      query: { ...scopeQuery(options), include_inactive: options.includeInactive, limit: options.limit },
    });
  }

  search(query: string, options: SearchOptions = {}, opts?: RequestOptions): Promise<SearchResult> {
    return this.request("POST", "/v1/search", {
      ...opts,
      body: compact({
        query,
        user_id: options.userId,
        agent_id: options.agentId,
        limit: options.limit,
        granularity: options.granularity,
        kinds: options.kinds,
        include_shared: options.includeShared,
        as_of: options.asOf,
      }),
    });
  }

  answer(question: string, options: AnswerOptions = {}, opts?: RequestOptions): Promise<Answer> {
    return this.request("POST", "/v1/answer", {
      ...opts,
      body: compact({
        question,
        user_id: options.userId,
        agent_id: options.agentId,
        limit: options.limit,
        style: options.style,
      }),
    });
  }

  /** Resolve a memory's evidence back to the raw episodes, with integrity flags. */
  evidence(memoryId: string, scope: Scope = {}, opts?: RequestOptions): Promise<ResolvedEvidence[]> {
    return this.request("GET", `/v1/memories/${enc(memoryId)}/evidence`, { ...opts, query: scopeQuery(scope) });
  }

  /** Version chain (oldest first) with ledger entries for each version. */
  history(memoryId: string, scope: Scope = {}, opts?: RequestOptions): Promise<HistoryEntry[]> {
    return this.request("GET", `/v1/memories/${enc(memoryId)}/history`, { ...opts, query: scopeQuery(scope) });
  }

  // ------------------------------------------------------------ outcomes
  /** Report how well a search/answer worked: reward in [-1, 1]. */
  feedback(traceId: string, reward: number, options: FeedbackOptions = {}, opts?: RequestOptions): Promise<FeedbackResult> {
    return this.request("POST", "/v1/feedback", {
      ...opts,
      body: compact({
        trace_id: traceId,
        reward,
        cited: options.cited,
        user_id: options.userId,
        agent_id: options.agentId,
      }),
    });
  }

  /** Distil a reusable experience from a task trajectory and its outcome. */
  learn(task: string, trajectory: unknown, reward: number, options: LearnOptions = {}, opts?: RequestOptions): Promise<LearnResult> {
    return this.request("POST", "/v1/learn", {
      ...opts,
      body: compact({
        task,
        trajectory,
        reward,
        user_id: options.userId,
        agent_id: options.agentId,
        scope: options.scope,
        shareable: options.shareable,
      }),
    });
  }

  consolidate(options: { userId?: string } = {}, opts?: RequestOptions): Promise<JsonObject> {
    return this.request("POST", "/v1/consolidate", { ...opts, query: { user_id: options.userId } });
  }

  /** May these memories bind a protected tool argument? Only Authorized ones may. */
  checkAction(memoryIds: string[], scope: Scope = {}, opts?: RequestOptions): Promise<CheckActionResult> {
    return this.request("POST", "/v1/check-action", {
      ...opts,
      body: compact({ memory_ids: memoryIds, user_id: scope.userId, agent_id: scope.agentId }),
    });
  }

  // ------------------------------------------------------------- sharing
  share(userId: string, options: ShareOptions = {}, opts?: RequestOptions): Promise<JsonObject> {
    return this.request("POST", "/v1/share", {
      ...opts,
      body: compact({
        user_id: userId,
        team_id: options.teamId,
        memory_ids: options.memoryIds,
        min_utility: options.minUtility,
      }),
    });
  }

  /** Memories shared with a user (team and public pool). */
  shared(options: { userId?: string; limit?: number } = {}, opts?: RequestOptions): Promise<MemoryItem[]> {
    return this.request("GET", "/v1/shared", { ...opts, query: { user_id: options.userId, limit: options.limit } });
  }

  /** A user's teams and the agents granted access to their memory. */
  teams(options: { userId?: string } = {}, opts?: RequestOptions): Promise<TeamsResult> {
    return this.request("GET", "/v1/teams", { ...opts, query: { user_id: options.userId } });
  }

  /** Low-level access-grant call; prefer the named helpers below. */
  grant(kind: GrantKind, subject: string, object: string, options: GrantOptions = {}, opts?: RequestOptions): Promise<GrantResult> {
    return this.request("POST", "/v1/grants", {
      ...opts,
      body: { kind, subject, object, revoke: options.revoke ?? false },
    });
  }

  revoke(kind: GrantKind, subject: string, object: string, opts?: RequestOptions): Promise<GrantResult> {
    return this.grant(kind, subject, object, { revoke: true }, opts);
  }

  /** Let `agentId` act on `userId`'s memories. */
  grantAgent(userId: string, agentId: string, opts?: RequestOptions): Promise<GrantResult> {
    return this.grant("user_agent", userId, agentId, {}, opts);
  }

  revokeAgent(userId: string, agentId: string, opts?: RequestOptions): Promise<GrantResult> {
    return this.revoke("user_agent", userId, agentId, opts);
  }

  /** Let `agentId` read memories derived from `resource`. */
  grantResource(agentId: string, resource: string, opts?: RequestOptions): Promise<GrantResult> {
    return this.grant("agent_resource", agentId, resource, {}, opts);
  }

  revokeResource(agentId: string, resource: string, opts?: RequestOptions): Promise<GrantResult> {
    return this.revoke("agent_resource", agentId, resource, opts);
  }

  joinTeam(userId: string, teamId: string, opts?: RequestOptions): Promise<GrantResult> {
    return this.grant("team_member", userId, teamId, {}, opts);
  }

  leaveTeam(userId: string, teamId: string, opts?: RequestOptions): Promise<GrantResult> {
    return this.revoke("team_member", userId, teamId, opts);
  }

  // ----------------------------------------------------------- integrity
  /** Check the ledger chain, episode hashes and memory text against the ledger. */
  verify(opts?: RequestOptions): Promise<VerifyResult> {
    return this.request("GET", "/v1/verify", { ...opts });
  }

  /**
   * Portable JSON of everything stored about a user (data-subject access).
   * With a per-user key, `userId` may be omitted (the key's user is used).
   */
  export(userId?: string, opts?: RequestOptions): Promise<ExportResult> {
    return this.request("GET", "/v1/export", { ...opts, query: { user_id: userId } });
  }

  // ------------------------------------------------------- admin: keys
  /** List API keys (metadata only, never secrets). Admin only. */
  listKeys(opts?: RequestOptions): Promise<ApiKeyInfo[]> {
    return this.request("GET", "/v1/keys", { ...opts });
  }

  /** Create an API key. The returned `key` is shown only once. Admin only. */
  createKey(options: CreateKeyOptions = {}, opts?: RequestOptions): Promise<CreatedKey> {
    return this.request("POST", "/v1/keys", {
      ...opts,
      body: compact({
        role: options.role,
        user_id: options.userId,
        agent_id: options.agentId,
        label: options.label,
      }),
    });
  }

  /** Revoke an API key by its prefix. Admin only. */
  revokeKey(prefix: string, opts?: RequestOptions): Promise<RevokeKeyResult> {
    return this.request("DELETE", `/v1/keys/${enc(prefix)}`, { ...opts });
  }

  // ------------------------------------------------------------ plumbing
  /** Send a raw request. Exposed for endpoints this client does not wrap yet. */
  async request<T = unknown>(
    method: string,
    path: string,
    init: RequestOptions & { query?: Query; body?: unknown } = {},
  ): Promise<T> {
    const url = this.baseUrl + path + buildQuery(init.query);
    const headers: Record<string, string> = { Accept: "application/json" };
    if (init.body !== undefined) headers["Content-Type"] = "application/json";
    if (this.apiKey) headers["Authorization"] = `Bearer ${this.apiKey}`;

    const controller = new AbortController();
    const timeoutMs = init.timeoutMs ?? this.timeoutMs;
    let timedOut = false;
    const timer =
      timeoutMs > 0
        ? setTimeout(() => {
            timedOut = true;
            controller.abort();
          }, timeoutMs)
        : undefined;
    const onAbort = () => controller.abort();
    if (init.signal) {
      if (init.signal.aborted) controller.abort();
      else init.signal.addEventListener("abort", onAbort, { once: true });
    }

    let res: Response;
    let text: string;
    try {
      res = await this.fetchImpl(url, {
        method,
        headers,
        body: init.body === undefined ? undefined : JSON.stringify(init.body),
        signal: controller.signal,
      });
      text = await res.text();
    } catch (err) {
      const detail = timedOut ? "timeout" : init.signal?.aborted ? "aborted" : errorMessage(err);
      const msg = timedOut
        ? `VeraMem ${method} ${path} timed out after ${timeoutMs} ms`
        : `VeraMem ${method} ${path} failed: ${detail}`;
      throw new VeraMemError(msg, { status: 0, detail, method, url, cause: err });
    } finally {
      if (timer !== undefined) clearTimeout(timer);
      init.signal?.removeEventListener("abort", onAbort);
    }

    let data: unknown = undefined;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = text;
      }
    }
    if (!res.ok) {
      const detail = isRecord(data) && "detail" in data ? data["detail"] : data;
      const shown = typeof detail === "string" ? detail : JSON.stringify(detail ?? res.statusText);
      throw new VeraMemError(`VeraMem ${method} ${path} -> ${res.status}: ${shown}`, {
        status: res.status,
        detail,
        method,
        url,
      });
    }
    return data as T;
  }
}

function enc(s: string): string {
  return encodeURIComponent(s);
}

function buildQuery(q?: Query): string {
  if (!q) return "";
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(q)) if (v !== undefined && v !== null) p.append(k, String(v));
  const s = p.toString();
  return s ? `?${s}` : "";
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
