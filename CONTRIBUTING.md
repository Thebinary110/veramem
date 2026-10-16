# Contributing to VeraMem

Thanks for your interest. VeraMem stores people's private memories, so it uses a
**vouched-contributor** model: every contributor is known to the maintainers before
their code is reviewed.

## How to contribute

1. **Open an issue first** using the *Proposal* template. Describe the problem and your
   planned change. Small typo fixes are welcome as issues too.
2. A maintainer discusses the proposal with you. If it is accepted, they add your GitHub
   username to [`.github/TRUSTED_CONTRIBUTORS`](.github/TRUSTED_CONTRIBUTORS).
3. Open your pull request. Pull requests from people who are not vouched for are
   closed automatically, with a link back here. That is not a judgement of the work.
4. CI must pass. For outside contributors, CI runs only after a maintainer approves the
   run. A code owner must approve the PR, and new pushes dismiss earlier approvals.
5. Commits on `main` must be signed. Set up signing once with
   `git config commit.gpgsign true` and an SSH or GPG key registered on GitHub.

## Development

```bash
git clone https://github.com/Thebinary110/veramem && cd veramem
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,mcp]"
pytest                                  # deterministic, no model or network needed
ruff check . && ruff format --check .
python -m benchmarks.guarantees --offline
```

## Ground rules

- No telemetry, analytics or "phone home" code, ever.
- New runtime dependencies need a strong reason, discussed in the issue first.
- Access checks happen in SQL before ranking. Never post-filter results.
- Nothing derived from a tool, a document, the assistant or another user may raise a
  memory's authority.
- Changes to retrieval, extraction or prompts must include benchmark numbers before and after.
- Be kind. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
