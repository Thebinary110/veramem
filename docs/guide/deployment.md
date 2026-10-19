# Deploying for many users

## Choose a backend

| | SQLite (default) | PostgreSQL |
|---|---|---|
| Setup | nothing: one file | `pip install "veramem[postgres]"`, `VERAMEM_DB=postgresql://...` |
| Processes | many readers, one writer at a time (WAL); several processes are safe | any number of processes and server replicas |
| Full-text search | FTS5 | generated `tsvector` + GIN index |
| Tested | the whole suite | the whole suite (CI runs it against PostgreSQL 16) |

Both backends keep the same guarantees:
- access checks run in SQL before ranking;
- the ledger hash chain stays linear across processes (`BEGIN IMMEDIATE` on SQLite, an advisory lock on PostgreSQL);
- cache invalidation is shared through the database, so a change made by one replica is seen by every other replica on its next search.

## Authenticate callers

With no authentication configured, the REST server trusts the `user_id` in each request,
and the CLI only allows that on `127.0.0.1`. For anything else, use one of the following.

**Per-user API keys** (recommended for agents and services). Keys are stored as SHA-256 hashes.

```bash
veramem keys create --admin --label ops        # can act for any user
veramem keys create -u alice --label laptop    # pinned to alice
veramem keys create -u alice --agent travel-bot
veramem keys list
veramem keys revoke vm_AbCdEf1
```

A user key can never act as another user. If a request names a different `user_id`, it
gets a 403. Admin keys can also be created and revoked from the web UI's Admin tab or
through `POST /v1/keys`.

**OIDC / JWT** (for your existing login):
- `VERAMEM_JWT_JWKS_URL=https://issuer/.well-known/jwks.json` verifies RS256, ES256 or EdDSA tokens.
- `VERAMEM_JWT_SECRET` verifies HS256 tokens.
- The `sub` claim becomes the user. `veramem_role: admin` makes the token an admin token.
- Set `VERAMEM_JWT_AUDIENCE` and `VERAMEM_JWT_ISSUER` to pin the token's audience and issuer.
- Requires `pip install "veramem[auth]"`.

**What users can and can't do with their own key:**
- They read and write their own memory.
- They grant and revoke which agents may use their memory.
- They share their consented experiences.
- They export or erase their own data.

Team membership, agent→resource grants, ledger verification and key management need an
admin.

## Production checklist

- [ ] TLS in front of VeraMem (reverse proxy or load balancer).
- [ ] An admin key or JWT configured; `veramem serve` refuses a public bind without one.
- [ ] `VERAMEM_RATE_LIMIT` set for your traffic (default 600 requests/min per caller).
- [ ] `VERAMEM_CORS_ORIGINS` set only if browsers call the API directly.
- [ ] Regular backups:
  - SQLite: `sqlite3 memory.db ".backup backup.db"`;
  - PostgreSQL: `pg_dump`.
- [ ] `veramem verify` (or `GET /v1/verify`) in monitoring. A failure means stored data was changed outside VeraMem.
- [ ] On PostgreSQL, `VACUUM FULL` scheduled if erased data must be physically rewritten out of table pages.

## Docker

```bash
docker run -d -p 127.0.0.1:8765:8765 -v veramem:/data \
  -e VERAMEM_API_KEY=$(openssl rand -hex 24) ghcr.io/thebinary110/veramem:latest
```

The image runs as an unprivileged user. Release images are multi-arch (amd64, arm64) and
carry SLSA build provenance:

```bash
gh attestation verify oci://ghcr.io/thebinary110/veramem:<version> --repo Thebinary110/veramem
```
