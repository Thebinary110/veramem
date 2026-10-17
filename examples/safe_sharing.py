"""Gap C: share what a user's agent learned with a team, without sharing the user."""

from veramem import Memory

mem = Memory("sharing.db")
mem.join_team("alice", "support")
mem.join_team("bob", "support")

mem.add("My account number is 4417-2290 and I live at 12 Baker Street", user_id="alice")

# Alice's agent solved a task; distil it into a reusable experience (needs an LLM).
out = mem.learn(
    "refund a duplicate charge",
    [
        "Looked up the two charges on the card ending 2290",
        "Confirmed with the customer which one to refund",
        "Issued the refund and sent the confirmation email",
    ],
    reward=1.0,
    user_id="alice",
    shareable=True,  # Alice consents to sharing de-identified experiences
)
print(out)

report = mem.share(user_id="alice", team_id="support")
print(report)  # rejected items show the PII / canary findings

for h in mem.search("how do I refund a duplicate charge?", user_id="bob").hits:
    print(h.granularity, h.memory.authority.value, h.memory.text[:120])
