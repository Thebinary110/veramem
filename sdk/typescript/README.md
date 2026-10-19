# veramem (TypeScript SDK)

A small TypeScript client for the [VeraMem](https://github.com/Thebinary110/veramem) REST API:
verifiable, outcome-aware, share-safe memory for AI agents.

- No runtime dependencies; uses the global `fetch` (Node >= 18, Deno, Bun, browsers)
- ESM and CommonJS builds with type declarations
- camelCase options mapped to the server's snake_case JSON
- Typed errors (`VeraMemError`) and per-request timeouts

## Install

```bash
npm install veramem
```

You also need a running VeraMem server (Python):

```bash
pip install veramem
veramem serve            # http://127.0.0.1:8765
```

## Quickstart (Node)

```ts
import { VeraMem } from "veramem";

const mem = new VeraMem({
  baseUrl: "http://127.0.0.1:8765",
  apiKey: process.env.VERAMEM_API_KEY, // omit for an open localhost server
});

await mem.add(
  [{ role: "user", content: "I moved to Lisbon last month and I'm vegetarian." }],
  { userId: "alice" },
);

const res = await mem.search("what should I cook for alice?", { userId: "alice", limit: 5 });
console.log(res.context); // compact, citable context block for your prompt

// After your agent acts, report how well the retrieved memories worked.
await mem.feedback(res.trace_id, 1, { cited: [res.hits[0]!.memory.id] });

const answer = await mem.answer("Where does alice live?", { userId: "alice" });
console.log(answer.answer, answer.citations);
```

CommonJS works too: `const { VeraMem } = require("veramem");`

## Quickstart (browser)

```html
<script type="module">
  import { VeraMem } from "https://cdn.jsdelivr.net/npm/veramem@0.1.0/dist/esm/index.js";

  // Use a per-user key here, never an admin key.
  const mem = new VeraMem({ baseUrl: "https://memory.example.com", apiKey: userKey });
  const { hits } = await mem.search("travel preferences");
</script>
```

The server has CORS off by default; set `VERAMEM_CORS_ORIGINS=https://your.app` to allow
browser calls from your origin.

## Authentication and `userId`

The client sends `Authorization: Bearer <apiKey>` when `apiKey` is set.

**The REST API trusts the `userId` / `agentId` you pass only when you use an admin key.**
A per-user key (created with `veramem keys create --user alice` or `mem.createKey(...)`)
is pinned to its user: requests act as that user automatically, and passing a different
`userId` returns 403. With a user key you can simply omit `userId`.

```ts
const admin = new VeraMem({ apiKey: process.env.VERAMEM_API_KEY });
const { key } = await admin.createKey({ role: "user", userId: "alice" }); // shown once

const alice = new VeraMem({ apiKey: key });
await alice.add("I prefer window seats.");   // stored for alice
await alice.me();                            // { role: "user", user_id: "alice", ... }
```

## API

All methods return promises and accept an optional last argument
`{ signal?: AbortSignal, timeoutMs?: number }`.

| Method | Endpoint |
| --- | --- |
| `health()` | `GET /health` |
| `info()`, `me()` | `GET /v1/info`, `GET /v1/me` |
| `add(messages, { userId, agentId, sessionId, teamId, infer, timestamp, metadata, resources })` | `POST /v1/memories` |
| `list({ userId, agentId, includeInactive, limit })` | `GET /v1/memories` |
| `get(id, scope)` / `update(id, text, scope)` | `GET` / `PUT /v1/memories/{id}` |
| `delete(id, { hard })` / `forget(id)` | `DELETE /v1/memories/{id}` (`forget` = hard delete) |
| `setShareable(id, shareable)` | `PUT /v1/memories/{id}/shareable` |
| `evidence(id)` / `history(id)` | `GET /v1/memories/{id}/evidence`, `/history` |
| `search(query, { limit, granularity, kinds, includeShared, asOf, ... })` | `POST /v1/search` |
| `answer(question, { limit, style, ... })` | `POST /v1/answer` |
| `feedback(traceId, reward, { cited })` | `POST /v1/feedback` |
| `learn(task, trajectory, reward, { scope, shareable, ... })` | `POST /v1/learn` |
| `share(userId, { teamId, memoryIds, minUtility })`, `shared()`, `teams()` | `/v1/share`, `/v1/shared`, `/v1/teams` |
| `grant(kind, subject, object)`, `revoke(...)`, `grantAgent`, `revokeAgent`, `grantResource`, `revokeResource`, `joinTeam`, `leaveTeam` | `POST /v1/grants` |
| `checkAction(memoryIds, scope)` | `POST /v1/check-action` |
| `consolidate({ userId })` | `POST /v1/consolidate` |
| `verify()` | `GET /v1/verify` |
| `export(userId?)` | `GET /v1/export` |
| `forgetUser(userId)` | `DELETE /v1/users/{id}` |
| `listKeys()`, `createKey(opts)`, `revokeKey(prefix)` | `/v1/keys` (admin) |
| `request(method, path, { query, body })` | any endpoint |

Response objects use the server's field names (snake_case), e.g. `res.trace_id`,
`hit.memory.valid_from`.

## Errors and timeouts

```ts
import { VeraMemError } from "veramem";

try {
  await mem.get("mem_missing");
} catch (e) {
  if (e instanceof VeraMemError) {
    console.log(e.status, e.detail); // 404 "not found: ..."
    if (e.isTimeout) { /* request exceeded timeoutMs */ }
  }
}
```

`status` is the HTTP status, or `0` for network failures, timeouts and aborts. The default
timeout is 60 s (`new VeraMem({ timeoutMs })`, or per call).

## Development

```bash
npm ci
npm run lint     # type-check
npm run build    # dist/esm + dist/cjs
npm test         # unit tests (mocked fetch)

# integration tests against a live server
VERAMEM_HOME=$(mktemp -d) VERAMEM_LLM=none VERAMEM_EMBEDDER=hash veramem serve --port 8797 &
VERAMEM_URL=http://127.0.0.1:8797 npm test
```

Tooling needs Node >= 20; the published package runs on Node >= 18.

## License

Apache-2.0
