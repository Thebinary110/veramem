/**
 * Live integration test. Runs only when VERAMEM_URL points at a running server, e.g.
 *   VERAMEM_HOME=$(mktemp -d) VERAMEM_LLM=none VERAMEM_EMBEDDER=hash veramem serve --port 8797
 *   VERAMEM_URL=http://127.0.0.1:8797 npm test
 * Set VERAMEM_API_KEY too if the server requires one.
 */
import { describe, expect, it } from "vitest";
import { VeraMem, VeraMemError } from "../src/index.js";

const url = process.env["VERAMEM_URL"];

describe.skipIf(!url)("live server", () => {
  const mem = new VeraMem({ baseUrl: url, apiKey: process.env["VERAMEM_API_KEY"] || undefined });
  const userId = `ts-sdk-${Date.now()}`;

  it("round-trips the core lifecycle", async () => {
    const health = await mem.health();
    expect(health.status).toBe("ok");

    const added = await mem.add(
      [
        { role: "user", content: "My favourite programming language is OCaml." },
        { role: "user", content: "I live in Lisbon and work as a data engineer." },
      ],
      { userId, infer: false },
    );
    expect(added.episodes.length).toBeGreaterThan(0);
    const ids = added.changes.filter((c) => c.op === "ADD" && c.memory_id).map((c) => c.memory_id!);
    expect(ids.length).toBeGreaterThan(0);

    const listed = await mem.list({ userId });
    expect(listed.length).toBeGreaterThan(0);

    const res = await mem.search("which programming language does the user like?", { userId, limit: 5 });
    expect(res.trace_id).toBeTruthy();
    expect(res.hits.length).toBeGreaterThan(0);
    expect(typeof res.context).toBe("string");
    const top = res.hits[0]!.memory;
    expect(top.user_id).toBe(userId);

    const fb = await mem.feedback(res.trace_id, 1, { cited: [top.id] });
    expect(fb.trace_id).toBe(res.trace_id);

    const ev = await mem.evidence(top.id, { userId });
    expect(ev.length).toBeGreaterThan(0);
    expect(ev[0]!.intact).toBe(true);

    const hist = await mem.history(top.id, { userId });
    expect(hist.length).toBeGreaterThan(0);
    expect(hist[hist.length - 1]!.id).toBe(top.id);
    expect(hist[0]!.ledger.length).toBeGreaterThan(0);

    const v = await mem.verify();
    expect(v.ok).toBe(true);

    const exp = await mem.export(userId);
    expect(exp.format).toBe("veramem.export.v1");
    expect(exp.memories.length).toBeGreaterThan(0);

    const gone = await mem.forgetUser(userId);
    expect(gone.memories).toBeGreaterThan(0);
    expect(await mem.list({ userId })).toEqual([]);
  });

  it("pins per-user keys to their user (admin key required)", async (ctx) => {
    const who = await mem.me();
    if (who.role !== "admin") return ctx.skip();
    const bob = `${userId}-bob`;
    const created = await mem.createKey({ role: "user", userId: bob, label: "ts-sdk-test" });
    try {
      const bobMem = new VeraMem({ baseUrl: url, apiKey: created.key });
      expect(await bobMem.me()).toMatchObject({ role: "user", user_id: bob });
      const added = await bobMem.add("I prefer window seats.", { infer: false });
      expect(added.episodes.length).toBe(1);
      const r = await bobMem.search("seat preference");
      expect(r.hits.every((h) => h.memory.user_id === bob)).toBe(true);
      const denied = (await bobMem.list({ userId }).catch((x: unknown) => x)) as VeraMemError;
      expect(denied).toBeInstanceOf(VeraMemError);
      expect(denied.status).toBe(403);
      await bobMem.forgetUser(bob);
    } finally {
      expect((await mem.revokeKey(created.prefix)).revoked).toBe(1);
    }
  });

  it("surfaces server errors as VeraMemError", async () => {
    const e = (await mem.get("mem_does_not_exist", { userId }).catch((x: unknown) => x)) as VeraMemError;
    expect(e).toBeInstanceOf(VeraMemError);
    expect(e.status).toBe(404);
  });
});
