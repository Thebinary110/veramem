# Security policy

## Reporting a vulnerability

**Do not open a public issue.** Use GitHub's private reporting:
<https://github.com/Thebinary110/veramem/security/advisories/new>.

You should get an acknowledgement within 3 days, and a fix or mitigation plan within
14 days for confirmed issues. We credit reporters in release notes unless you prefer
otherwise.

## Supported versions

Only the latest minor release receives security fixes.

## What VeraMem guarantees

These guarantees are covered by tests in `tests/` and by `benchmarks/guarantees.py`, which
CI runs on every change:

| Guarantee | How it is enforced |
|---|---|
| A user never retrieves another user's private memory | The access predicate is compiled into SQL *before* ranking (`policy.read_filter`), so unreadable rows never reach the scorer. |
| Dedup never looks across owners | Reconciliation compares only against memories in the same scope (`write._scope_where`). |
| Revoking an agent or resource grant takes effect immediately and retroactively | Grants are evaluated at read time. |
| Preferences are never shared | Invariant checked in `set_shareable` and `sharing`. |
| Shared experiences carry no personal data | A deterministic PII and canary gate runs after LLM abstraction (`privacy.check_shareable`). |
| Tool, document and assistant claims never become user facts, preferences or decisions | Authority comes from the source role and can only fall during consolidation. |
| Low-authority memories cannot drive protected actions | `check_action` returns `needs_confirmation`. |
| History cannot be silently rewritten | Writes go to a SHA-256 hash-chained ledger. `veramem verify` also re-hashes episodes and memory text. |
| Erasure is real | `forget` deletes content. The ledger keeps only hashes. |

## Out of scope

- An attacker with write access to the SQLite file. The ledger *detects* tampering but
  cannot prevent it. Protect the file with OS permissions or disk encryption.
- Prompt injection *inside* a user's own messages affecting that same user's memory.
  Authority labels limit its effect on actions; they do not stop extraction.
- The REST server's `user_id` parameter is trusted. Put VeraMem behind your own
  authentication when exposing it to end users. Never bind it to a public interface
  without `VERAMEM_API_KEY`; the CLI refuses to do so.

## Supply chain

- Releases are built in GitHub Actions and attested with SLSA build provenance.
  - Verify with `gh attestation verify <file> --repo Thebinary110/veramem`.
  - Artifacts are also signed with Sigstore.
- Releases are published to PyPI through OIDC trusted publishing. No long-lived token exists.
- Every third-party GitHub Action is pinned to a full commit SHA. Dependabot proposes updates.
- VeraMem collects **no telemetry** and makes no network calls except to the model
  provider you configure.
