"""Concurrency: many threads adding, searching and rating at once."""

import asyncio
import concurrent.futures as cf
import random

from veramem import AsyncMemory
from veramem.providers import HashEmbedder, NoLLM


def test_threads_keep_ledger_and_isolation_intact(raw_mem):
    errors = []

    def work(i):
        rng = random.Random(i)
        try:
            for j in range(15):
                u = f"u{rng.randint(0, 4)}"
                if rng.random() < 0.5:
                    raw_mem.add(f"secret-{u} fact {i}-{j}", user_id=u)
                else:
                    r = raw_mem.search("fact", user_id=u)
                    assert all(h.memory.user_id == u for h in r.hits)
                    if r.hits:
                        raw_mem.feedback(r.trace_id, rng.choice([1, -1]))
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    with cf.ThreadPoolExecutor(8) as ex:
        list(ex.map(work, range(24)))
    assert errors == []
    assert raw_mem.verify()["ok"]


def test_async_facade(tmp_path):
    async def run():
        async with AsyncMemory(str(tmp_path / "a.db"), llm=NoLLM(), embedder=HashEmbedder()) as m:
            await asyncio.gather(*(m.add(f"note {i} about tea", user_id="z") for i in range(10)))
            r = await m.search("tea", user_id="z")
            assert len(r.hits) == 10
            assert (await m.verify())["ok"]

    asyncio.run(run())
