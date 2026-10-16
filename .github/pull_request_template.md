## What and why

<!-- Link the issue this implements. PRs without a prior issue are closed. -->
Closes #

## How it was tested

- [ ] `pytest` passes locally
- [ ] Added or updated tests for the change
- [ ] If retrieval or extraction changed: ran `python -m benchmarks.guarantees` and attached numbers

## Safety checklist

- [ ] No new network calls, telemetry, or dependencies without discussion
- [ ] Access checks still happen before ranking (`policy.read_filter`)
- [ ] Nothing derived from a tool, document, or another user can raise authority
