# Governance

## Roles

- **Maintainers** (listed in `.github/CODEOWNERS`) review and merge, cut releases, and
  manage `.github/TRUSTED_CONTRIBUTORS`.
- **Trusted contributors** may open pull requests.
- **Everyone** may open issues, proposals and discussions.

A trusted contributor with a record of good contributions may be nominated as a
maintainer by an existing maintainer. Maintainers decide by consensus.

## How changes reach users

```
proposal issue -> maintainer vouches -> PR -> CI (approved run) -> code-owner review
  -> squash merge to protected main (signed, linear) -> signed tag by a maintainer
  -> release workflow (tests, build, SLSA provenance) -> manual approval in `pypi` env
  -> PyPI via OIDC trusted publishing -> Sigstore-signed GitHub release
```

## Repository settings

Applied by `scripts/harden-github.sh`, and listed here so they can be audited:

- **`main` ruleset:** no deletion, no force push, linear history, signed commits.
  - PR required: 1 approval, code-owner review, stale reviews dismissed, last-push approval, conversations resolved.
  - Squash merge only.
  - Required checks: lint, test, build, CodeQL, secret scan, dependency audit, contributor gate.
  - No bypass actors.
- **Tag ruleset:** only repository admins can create or move `v*` tags, and tags must be signed.
- **Actions:**
  - The default token is read-only and can't approve PRs.
  - Only GitHub-owned and explicitly allow-listed actions can run, and they must be SHA-pinned.
  - Runs from every external contributor need maintainer approval.
- **`pypi` environment:** a maintainer must approve each deployment, and it only runs from `v*` tags.
- **Security features:** secret scanning with push protection, Dependabot alerts and fixes, and private vulnerability reporting.
