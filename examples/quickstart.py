"""60-second tour. Run: python examples/quickstart.py (uses Ollama/OpenAI/Anthropic if found)."""

from veramem import Memory

mem = Memory("quickstart.db")
print("using", mem.info()["llm"])

mem.add(
    [
        {"role": "user", "content": "I'm Priya. I moved to Lisbon last week for a job at Feedzai."},
        {"role": "tool", "name": "crm", "content": "customer record: home_city=Porto"},
        {"role": "user", "content": "I'm vegetarian and allergic to peanuts."},
    ],
    user_id="priya",
)

# Every memory knows where it came from, and who said it.
for m in mem.get_all(user_id="priya"):
    print(f"[{m.authority.value:10}] {m.text}")

# Answers cite memories, and abstain instead of guessing.
print(mem.answer("Where does Priya live?", user_id="priya").answer)
print(mem.answer("What is Priya's passport number?", user_id="priya").answer)

# Knowledge changes are versioned, not overwritten.
mem.add("Update: I've moved to Madrid now.", user_id="priya")
print(mem.answer("Where does Priya live now?", user_id="priya").answer)

# Tamper-evident: every write is in a hash chain.
print("ledger ok:", mem.verify()["ok"])
