/**
 * Types mirroring VeraMem's server data model (src/veramem/types.py).
 * Wire format is the server's snake_case JSON; timestamps are Unix seconds (float).
 */

/** Who produced a piece of raw evidence. */
export type SourceType = "user" | "assistant" | "tool" | "document" | "system" | "shared";

/** How much a memory may be trusted to act on. */
export type Authority = "authorized" | "attested" | "unendorsed";

export type Kind = "fact" | "preference" | "event" | "experience" | "summary";

export type Status = "active" | "superseded" | "retracted" | "archived";

export type Visibility = "private" | "team" | "public";

export type Operation = "ADD" | "UPDATE" | "DELETE" | "NOOP";

export type Granularity = "auto" | "fact" | "summary" | "raw" | "dual" | (string & {});

export type GrantKind = "user_agent" | "agent_resource" | "team_member";

/** A pointer into an Episode: the exact span that supports a memory. */
export interface Evidence {
  episode_id: string;
  quote: string;
  start: number;
  end: number;
  source: SourceType;
}

export interface MemoryItem {
  id: string;
  text: string;
  kind: Kind;
  user_id: string | null;
  agent_id: string | null;
  team_id: string | null;
  visibility: Visibility;
  authority: Authority;
  evidence: Evidence[];
  entities: string[];
  status: Status;
  supersedes: string | null;
  superseded_by: string | null;
  valid_from: number | null;
  valid_to: number | null;
  created_at: number;
  updated_at: number;
  confidence: number;
  uses: number;
  successes: number;
  failures: number;
  last_used_at: number | null;
  shareable: boolean;
  metadata: Record<string, unknown>;
}

/** Immutable raw evidence: one message, tool result or document chunk. */
export interface Episode {
  id: string;
  content: string;
  source: SourceType;
  actor: string | null;
  user_id: string | null;
  agent_id: string | null;
  session_id: string | null;
  occurred_at: number;
  ingested_at: number;
  sha256: string;
  metadata: Record<string, unknown>;
}

/** One retrieval result, with the reasons it was ranked where it is. */
export interface Hit {
  memory: MemoryItem;
  score: number;
  signals: Record<string, number>;
  granularity: string;
}

export interface SearchResult {
  trace_id: string;
  query: string;
  hits: Hit[];
  granularity: string;
  /** Compact, citable context block rendered by the server, ready for a prompt. */
  context: string;
}

export interface Change {
  op: Operation;
  memory_id: string | null;
  text: string | null;
  previous: string | null;
  reason: string;
}

export interface AddResult {
  episodes: string[];
  changes: Change[];
  /** Candidate memories rejected because they were not grounded in the input. */
  rejected: Record<string, unknown>[];
}

export interface Answer {
  trace_id: string;
  answer: string;
  citations: string[];
  abstained: boolean;
  used_memories: string[];
}

/** A memory's evidence pointer resolved back to its raw episode. */
export interface ResolvedEvidence {
  episode_id: string;
  quote: string;
  source: SourceType;
  span: [number, number];
  /** Raw episode text; null when it belongs to a different user or is missing. */
  episode: string | null;
  occurred_at: number | null;
  /** True when the episode still matches its stored hash. */
  intact: boolean;
}

export interface LedgerEntry {
  seq: number;
  ts: number;
  actor: string | null;
  op: string;
  payload: Record<string, unknown>;
  hash: string;
}

/** One version in a memory's history chain (oldest first). */
export interface HistoryEntry {
  id: string;
  text: string;
  status: Status;
  authority: Authority;
  valid_from: number | null;
  valid_to: number | null;
  ledger: LedgerEntry[];
}

export interface FeedbackResult {
  trace_id: string;
  reward: number;
  /** Memory ids that received credit for this outcome. */
  credited: string[];
  [key: string]: unknown;
}

export interface LearnResult {
  episodes: string[];
  action: string;
  experience: string | null;
  reason?: string;
  [key: string]: unknown;
}

export interface ForgetResult {
  memories: number;
  episodes: number;
  shared: number;
}

export interface RetractResult {
  retracted: string;
}

export interface GrantResult {
  status: "granted" | "revoked";
}

export interface CheckActionResult {
  allowed: boolean;
  needs_confirmation: string[];
  missing: string[];
}

export interface VerifyResult {
  ok: boolean;
  ledger_ok: boolean;
  ledger_entries: number;
  head: string;
  first_bad_seq: number | null;
  tampered_episodes: string[];
  tampered_memories: string[];
}

export interface ExportResult {
  format: "veramem.export.v1";
  user_id: string;
  exported_at: number;
  episodes: Episode[];
  memories: MemoryItem[];
}

export interface HealthResult {
  status: string;
  version: string;
  auth: boolean;
  [key: string]: unknown;
}

export interface InfoResult {
  db: string;
  llm: string;
  embedder: string;
  episodes: number;
  memories_active: number;
  memories_total: number;
  traces: number;
  ledger_entries: number;
  [key: string]: unknown;
}

export interface MeResult {
  role: "admin" | "user";
  user_id: string | null;
  agent_id: string | null;
}

export interface ShareableResult {
  id: string;
  shareable: boolean;
}

export interface TeamsResult {
  user_id: string | null;
  teams: string[];
  /** Agents granted access to this user's memory. */
  agents: string[];
}

export type KeyRole = "admin" | "user";

export interface ApiKeyInfo {
  prefix: string;
  role: KeyRole;
  user_id: string | null;
  agent_id: string | null;
  label: string | null;
  created_at: number;
  revoked_at: number | null;
  last_used_at: number | null;
}

export interface CreatedKey {
  /** The full secret. It is shown only once. */
  key: string;
  prefix: string;
  note: string;
}

export interface RevokeKeyResult {
  revoked: number;
}

/** Results whose shape depends on server configuration (share, consolidate). */
export type JsonObject = Record<string, unknown>;

// ------------------------------------------------------------------ inputs

/** A chat message as accepted by `add`. */
export interface Message {
  role: string;
  content: string;
  name?: string;
  [key: string]: unknown;
}

/** Principal for scoped calls. The server only trusts these with an admin key. */
export interface Scope {
  userId?: string;
  agentId?: string;
}

export interface AddOptions extends Scope {
  sessionId?: string;
  teamId?: string;
  /** Extract memories with the LLM (default true). False stores raw turns. */
  infer?: boolean;
  /** Event time: Unix seconds or an ISO-8601 string. */
  timestamp?: number | string;
  metadata?: Record<string, unknown>;
  resources?: string[];
}

export interface SearchOptions extends Scope {
  limit?: number;
  granularity?: Granularity;
  kinds?: Kind[];
  includeShared?: boolean;
  /** Time-travel: search the memory state as of this Unix timestamp (seconds). */
  asOf?: number;
}

export interface AnswerOptions extends Scope {
  limit?: number;
  style?: string;
}

export interface FeedbackOptions extends Scope {
  /** Memory ids your agent actually used (must come from that trace). */
  cited?: string[];
}

export interface ListOptions extends Scope {
  includeInactive?: boolean;
  limit?: number;
}

export interface DeleteOptions extends Scope {
  /** Permanently erase instead of retracting (kept for audit). */
  hard?: boolean;
}

export interface LearnOptions extends Scope {
  scope?: string;
  shareable?: boolean;
}

export interface ShareOptions {
  teamId?: string;
  memoryIds?: string[];
  minUtility?: number;
}

export interface CreateKeyOptions {
  role?: KeyRole;
  userId?: string;
  agentId?: string;
  label?: string;
}

export interface GrantOptions {
  revoke?: boolean;
}
