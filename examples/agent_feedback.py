"""Gap A: memories that help get ranked higher, memories that mislead get archived."""

from veramem import Memory

mem = Memory("feedback.db")
mem.add(
    [
        "The staging database password was rotated, ask ops for the new one",
        "Staging DB credentials: see vault path secret/staging/db",
    ],
    user_id="ops-bot",
    infer=False,  # store verbatim for this demo
)

for attempt in range(4):
    hit = mem.search("staging database password", user_id="ops-bot", limit=1)
    top = hit.hits[0].memory
    helped = "vault" in top.text  # your agent decides this from the task outcome
    print(f"attempt {attempt}: {top.text[:60]!r} -> {'success' if helped else 'failure'}")
    mem.feedback(hit.trace_id, 1.0 if helped else -1.0)

print(mem.consolidate(user_id="ops-bot", summarize=False))
