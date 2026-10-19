import { describe, expect, it, vi } from "vitest";
import { VeraMem, VeraMemError } from "../src/index.js";

type Call = { url: string; method: string; headers: Record<string, string>; body: unknown };

function mockFetch(responder: (call: Call) => { status?: number; body?: unknown } = () => ({ body: {} })) {
  const calls: Call[] = [];
  const fn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const call: Call = {
      url: String(input),
      method: init?.method ?? "GET",
      headers: (init?.headers ?? {}) as Record<string, string>,
      body: typeof init?.body === "string" ? JSON.parse(init.body) : undefined,
    };
    calls.push(call);
    const r = responder(call);
    const text = r.body === undefined ? "" : typeof r.body === "string" ? r.body : JSON.stringify(r.body);
    return new Response(text, { status: r.status ?? 200, headers: { "Content-Type": "application/json" } });
  });
  return { fetch: fn as unknown as typeof fetch, calls };
}

const BASE = "http://mem.test:9000";

function client(responder?: Parameters<typeof mockFetch>[0], apiKey?: string) {
  const m = mockFetch(responder);
  return { mem: new VeraMem({ baseUrl: BASE + "/", apiKey, fetch: m.fetch }), calls: m.calls };
}

describe("request mapping", () => {
  it("add maps camelCase options to snake_case body", async () => {
    const { mem, calls } = client(() => ({ body: { episodes: ["ep_1"], changes: [], rejected: [] } }));
    const r = await mem.add([{ role: "user", content: "I live in Berlin" }], {
      userId: "alice",
      agentId: "bot",
      sessionId: "s1",
      teamId: "t1",
      infer: false,
      timestamp: 1700000000,
      metadata: { a: 1 },
      resources: ["doc:1"],
    });
    expect(r.episodes).toEqual(["ep_1"]);
    expect(calls[0]).toMatchObject({ url: `${BASE}/v1/memories`, method: "POST" });
    expect(calls[0]!.body).toEqual({
      messages: [{ role: "user", content: "I live in Berlin" }],
      user_id: "alice",
      agent_id: "bot",
      session_id: "s1",
      team_id: "t1",
      infer: false,
      timestamp: 1700000000,
      metadata: { a: 1 },
      resources: ["doc:1"],
    });
    expect(calls[0]!.headers["Content-Type"]).toBe("application/json");
  });

  it("omits undefined fields so server defaults apply", async () => {
    const { mem, calls } = client();
    await mem.add("hello");
    expect(calls[0]!.body).toEqual({ messages: "hello" });
  });

  it("search / answer bodies", async () => {
    const { mem, calls } = client();
    await mem.search("where?", {
      userId: "u",
      agentId: "a",
      limit: 5,
      granularity: "fact",
      kinds: ["fact", "preference"],
      includeShared: false,
      asOf: 123.5,
    });
    await mem.answer("why?", { userId: "u", limit: 7, style: "detailed" });
    expect(calls[0]).toMatchObject({ url: `${BASE}/v1/search`, method: "POST" });
    expect(calls[0]!.body).toEqual({
      query: "where?",
      user_id: "u",
      agent_id: "a",
      limit: 5,
      granularity: "fact",
      kinds: ["fact", "preference"],
      include_shared: false,
      as_of: 123.5,
    });
    expect(calls[1]).toMatchObject({ url: `${BASE}/v1/answer`, method: "POST" });
    expect(calls[1]!.body).toEqual({ question: "why?", user_id: "u", limit: 7, style: "detailed" });
  });

  it("feedback sends trace_id, reward and cited", async () => {
    const { mem, calls } = client();
    await mem.feedback("tr_1", 1, { cited: ["mem_a"] });
    await mem.feedback("tr_2", -0.5);
    await mem.feedback("tr_3", 0, { userId: "u", agentId: "a" });
    expect(calls[0]!.body).toEqual({ trace_id: "tr_1", reward: 1, cited: ["mem_a"] });
    expect(calls[1]!.body).toEqual({ trace_id: "tr_2", reward: -0.5 });
    expect(calls[2]!.body).toEqual({ trace_id: "tr_3", reward: 0, user_id: "u", agent_id: "a" });
  });

  it("memory CRUD routes and query strings", async () => {
    const { mem, calls } = client();
    await mem.get("mem/1", { userId: "u" });
    await mem.list({ userId: "u", includeInactive: true, limit: 3 });
    await mem.update("mem_1", "new text", { agentId: "a" });
    await mem.delete("mem_1", { userId: "u" });
    await mem.delete("mem_1", { userId: "u", hard: true });
    await mem.forget("mem_2");
    await mem.evidence("mem_1", { userId: "u" });
    await mem.history("mem_1");
    expect(calls.map((c) => `${c.method} ${c.url}`)).toEqual([
      `GET ${BASE}/v1/memories/mem%2F1?user_id=u`,
      `GET ${BASE}/v1/memories?user_id=u&include_inactive=true&limit=3`,
      `PUT ${BASE}/v1/memories/mem_1`,
      `DELETE ${BASE}/v1/memories/mem_1?user_id=u`,
      `DELETE ${BASE}/v1/memories/mem_1?user_id=u&hard=true`,
      `DELETE ${BASE}/v1/memories/mem_2?hard=true`,
      `GET ${BASE}/v1/memories/mem_1/evidence?user_id=u`,
      `GET ${BASE}/v1/memories/mem_1/history`,
    ]);
    expect(calls[2]!.body).toEqual({ text: "new text", agent_id: "a" });
    expect(calls[0]!.body).toBeUndefined();
    expect(calls[0]!.headers["Content-Type"]).toBeUndefined();
  });

  it("learn, share, grants, check-action, consolidate", async () => {
    const { mem, calls } = client();
    await mem.learn("book flight", [{ role: "user", content: "x" }], 0.9, { userId: "u", scope: "travel", shareable: true });
    await mem.share("u", { teamId: "t", memoryIds: ["m1"], minUtility: 0.7 });
    await mem.grantAgent("u", "bot");
    await mem.revokeAgent("u", "bot");
    await mem.grantResource("bot", "doc:1");
    await mem.joinTeam("u", "t");
    await mem.leaveTeam("u", "t");
    await mem.checkAction(["m1", "m2"], { userId: "u" });
    await mem.consolidate({ userId: "u" });
    expect(calls[0]!.body).toEqual({
      task: "book flight",
      trajectory: [{ role: "user", content: "x" }],
      reward: 0.9,
      user_id: "u",
      scope: "travel",
      shareable: true,
    });
    expect(calls[1]!.body).toEqual({ user_id: "u", team_id: "t", memory_ids: ["m1"], min_utility: 0.7 });
    expect(calls.slice(2, 7).map((c) => c.body)).toEqual([
      { kind: "user_agent", subject: "u", object: "bot", revoke: false },
      { kind: "user_agent", subject: "u", object: "bot", revoke: true },
      { kind: "agent_resource", subject: "bot", object: "doc:1", revoke: false },
      { kind: "team_member", subject: "u", object: "t", revoke: false },
      { kind: "team_member", subject: "u", object: "t", revoke: true },
    ]);
    expect(calls.slice(2, 7).every((c) => c.url === `${BASE}/v1/grants`)).toBe(true);
    expect(calls[7]).toMatchObject({ url: `${BASE}/v1/check-action`, body: { memory_ids: ["m1", "m2"], user_id: "u" } });
    expect(calls[8]).toMatchObject({ method: "POST", url: `${BASE}/v1/consolidate?user_id=u` });
  });

  it("service, integrity and portability routes", async () => {
    const { mem, calls } = client();
    await mem.health();
    await mem.info();
    await mem.verify();
    await mem.export("a b");
    await mem.forgetUser("alice");
    expect(calls.map((c) => `${c.method} ${c.url}`)).toEqual([
      `GET ${BASE}/health`,
      `GET ${BASE}/v1/info`,
      `GET ${BASE}/v1/verify`,
      `GET ${BASE}/v1/export?user_id=a+b`,
      `DELETE ${BASE}/v1/users/alice`,
    ]);
  });
});

describe("newer endpoints", () => {
  it("me, shareable, shared, teams, keys", async () => {
    const { mem, calls } = client();
    await mem.me();
    await mem.setShareable("mem_1", false, { userId: "u" });
    await mem.shared({ userId: "u", limit: 5 });
    await mem.teams({ userId: "u" });
    await mem.listKeys();
    await mem.createKey({ role: "user", userId: "u", label: "ci" });
    await mem.revokeKey("vm_abc");
    await mem.export();
    expect(calls.map((c) => `${c.method} ${c.url}`)).toEqual([
      `GET ${BASE}/v1/me`,
      `PUT ${BASE}/v1/memories/mem_1/shareable`,
      `GET ${BASE}/v1/shared?user_id=u&limit=5`,
      `GET ${BASE}/v1/teams?user_id=u`,
      `GET ${BASE}/v1/keys`,
      `POST ${BASE}/v1/keys`,
      `DELETE ${BASE}/v1/keys/vm_abc`,
      `GET ${BASE}/v1/export`,
    ]);
    expect(calls[1]!.body).toEqual({ shareable: false, user_id: "u" });
    expect(calls[5]!.body).toEqual({ role: "user", user_id: "u", label: "ci" });
  });
});

describe("auth", () => {
  it("sends a bearer token when apiKey is set", async () => {
    const { mem, calls } = client(undefined, "sk-123");
    await mem.info();
    expect(calls[0]!.headers["Authorization"]).toBe("Bearer sk-123");
  });

  it("sends no Authorization header without a key", async () => {
    const { mem, calls } = client();
    await mem.info();
    expect(calls[0]!.headers["Authorization"]).toBeUndefined();
  });
});

describe("errors", () => {
  it("throws VeraMemError with status and detail", async () => {
    const { mem } = client(() => ({ status: 403, body: { detail: "agent bot may not read alice" } }));
    const err = await mem.get("mem_1").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(VeraMemError);
    expect(err).toBeInstanceOf(Error);
    const e = err as VeraMemError;
    expect(e.status).toBe(403);
    expect(e.detail).toBe("agent bot may not read alice");
    expect(e.method).toBe("GET");
    expect(e.message).toContain("403");
  });

  it("keeps structured validation details (422)", async () => {
    const detail = [{ loc: ["body", "reward"], msg: "too big" }];
    const { mem } = client(() => ({ status: 422, body: { detail } }));
    const e = (await mem.feedback("t", 5).catch((x: unknown) => x)) as VeraMemError;
    expect(e.status).toBe(422);
    expect(e.detail).toEqual(detail);
  });

  it("handles non-JSON error bodies", async () => {
    const { mem } = client(() => ({ status: 502, body: "Bad Gateway" }));
    const e = (await mem.health().catch((x: unknown) => x)) as VeraMemError;
    expect(e.status).toBe(502);
    expect(e.detail).toBe("Bad Gateway");
  });

  it("wraps network failures with status 0", async () => {
    const fetch = (async () => {
      throw new TypeError("fetch failed");
    }) as unknown as typeof globalThis.fetch;
    const mem = new VeraMem({ baseUrl: BASE, fetch });
    const e = (await mem.health().catch((x: unknown) => x)) as VeraMemError;
    expect(e).toBeInstanceOf(VeraMemError);
    expect(e.status).toBe(0);
    expect(e.detail).toBe("fetch failed");
    expect(e.isTimeout).toBe(false);
  });
});

describe("timeout", () => {
  const hanging = ((_url: unknown, init?: RequestInit) =>
    new Promise((_resolve, reject) => {
      init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
    })) as unknown as typeof fetch;

  it("aborts after timeoutMs", async () => {
    const mem = new VeraMem({ baseUrl: BASE, fetch: hanging, timeoutMs: 20 });
    const e = (await mem.health().catch((x: unknown) => x)) as VeraMemError;
    expect(e).toBeInstanceOf(VeraMemError);
    expect(e.isTimeout).toBe(true);
    expect(e.message).toContain("timed out after 20 ms");
  });

  it("per-request timeout overrides the client default", async () => {
    const mem = new VeraMem({ baseUrl: BASE, fetch: hanging, timeoutMs: 60000 });
    const e = (await mem.info({ timeoutMs: 10 }).catch((x: unknown) => x)) as VeraMemError;
    expect(e.isTimeout).toBe(true);
  });

  it("honours a caller AbortSignal", async () => {
    const mem = new VeraMem({ baseUrl: BASE, fetch: hanging });
    const ac = new AbortController();
    const p = mem.info({ signal: ac.signal }).catch((x: unknown) => x);
    ac.abort();
    const e = (await p) as VeraMemError;
    expect(e.status).toBe(0);
    expect(e.detail).toBe("aborted");
    expect(e.isTimeout).toBe(false);
  });

  it("does not time out a fast response", async () => {
    const { fetch } = mockFetch(() => ({ body: { status: "ok", version: "x", auth: false } }));
    const mem = new VeraMem({ baseUrl: BASE, fetch, timeoutMs: 1000 });
    await expect(mem.health()).resolves.toMatchObject({ status: "ok" });
  });
});

describe("constructor", () => {
  it("defaults to the local server", () => {
    const mem = new VeraMem({ fetch: mockFetch().fetch });
    expect(mem.baseUrl).toBe("http://127.0.0.1:8765");
    expect(mem.timeoutMs).toBe(60000);
  });
});
