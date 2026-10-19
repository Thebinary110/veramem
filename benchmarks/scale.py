"""Scale benchmark: ingest throughput and search latency. No model needed.

    python -m benchmarks.scale                  # single user up to 100k memories
    python -m benchmarks.scale --tenants 50     # 50 users x 2k memories in one store

Uses raw-turn ingestion (no LLM) and 768-d hash embeddings, so the numbers measure
VeraMem's storage and retrieval, not a model.
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import tempfile
import time

from veramem import Memory
from veramem.providers import HashEmbedder, NoLLM

VOCAB = [f"w{i}" for i in range(5000)]


def _text(rng: random.Random) -> str:
    return " ".join(rng.choices(VOCAB, k=12))


def _latency(mem: Memory, user: str, rng: random.Random, n: int = 30) -> tuple[float, float]:
    lat = []
    for _ in range(n):
        q = " ".join(rng.choices(VOCAB, k=5))
        t = time.perf_counter()
        mem.search(q, user_id=user, limit=10)
        lat.append(time.perf_counter() - t)
    lat.sort()
    return statistics.median(lat) * 1000, lat[int(0.95 * (n - 1))] * 1000


def single(sizes: list[int]) -> None:
    rng = random.Random(1)
    d = tempfile.mkdtemp()
    mem = Memory(os.path.join(d, "s.db"), llm=NoLLM(), embedder=HashEmbedder(dim=768))
    done = 0
    print("| memories | ingest (s) | search p50 (ms) | search p95 (ms) | db size (MB) |")
    print("|---:|---:|---:|---:|---:|")
    for target in sizes:
        t = time.time()
        while done < target:
            n = min(500, target - done)
            mem.add([_text(rng) for _ in range(n)], user_id="u", infer=False)
            done += n
        ingest = time.time() - t
        p50, p95 = _latency(mem, "u", rng)
        size = os.path.getsize(os.path.join(d, "s.db")) / 1e6
        print(f"| {target:,} | {ingest:.1f} | {p50:.1f} | {p95:.1f} | {size:.0f} |", flush=True)


def compare_index(sizes: list[int]) -> None:
    """Exact vs HNSW on the same database: latency and top-10 agreement."""
    rng = random.Random(3)
    d = tempfile.mkdtemp()
    path = os.path.join(d, "c.db")
    writer = Memory(path, llm=NoLLM(), embedder=HashEmbedder(dim=768))
    done = 0
    print(
        "| memories | exact p50 (ms) | hnsw p50 (ms) | hnsw first-search build (s) | top-10 overlap |"
    )
    print("|---:|---:|---:|---:|---:|")
    for target in sizes:
        while done < target:
            n = min(1000, target - done)
            writer.add([_text(rng) for _ in range(n)], user_id="u", infer=False)
            done += n
        exact = Memory(path, llm=NoLLM(), embedder=HashEmbedder(dim=768), vector_index="exact")
        ann = Memory(path, llm=NoLLM(), embedder=HashEmbedder(dim=768), vector_index="hnsw")
        t = time.time()
        ann.search("warm up", user_id="u", record=False)
        build = time.time() - t
        exact.search("warm up", user_id="u", record=False)
        qs = [" ".join(rng.choices(VOCAB, k=5)) for _ in range(30)]
        lat_e, lat_a, overlap = [], [], []
        for q in qs:
            t = time.perf_counter()
            e = exact.search(q, user_id="u", limit=10, record=False, include_raw=False).hits
            lat_e.append(time.perf_counter() - t)
            t = time.perf_counter()
            a = ann.search(q, user_id="u", limit=10, record=False, include_raw=False).hits
            lat_a.append(time.perf_counter() - t)
            overlap.append(len({h.memory.id for h in e} & {h.memory.id for h in a}) / 10)
        print(
            f"| {target:,} | {statistics.median(lat_e) * 1000:.1f} | "
            f"{statistics.median(lat_a) * 1000:.1f} | {build:.1f} | "
            f"{statistics.mean(overlap):.2f} |",
            flush=True,
        )
        exact.close()
        ann.close()


def tenants(n_users: int, per_user: int) -> None:
    rng = random.Random(2)
    d = tempfile.mkdtemp()
    mem = Memory(os.path.join(d, "t.db"), llm=NoLLM(), embedder=HashEmbedder(dim=768))
    t = time.time()
    for u in range(n_users):
        mem.add([_text(rng) for _ in range(per_user)], user_id=f"user{u}", infer=False)
    total = n_users * per_user
    print(f"ingested {total:,} memories for {n_users} users in {time.time() - t:.1f}s")
    p50, p95 = _latency(mem, "user7", rng)
    print(f"search for one user: p50 {p50:.1f} ms, p95 {p95:.1f} ms")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenants", type=int, default=0)
    ap.add_argument("--per-user", type=int, default=2000)
    ap.add_argument("--sizes", default="10000,50000,100000")
    ap.add_argument("--compare-index", action="store_true", help="exact vs HNSW")
    a = ap.parse_args()
    if a.compare_index:
        compare_index([int(x) for x in a.sizes.split(",")])
    elif a.tenants:
        tenants(a.tenants, a.per_user)
    else:
        single([int(x) for x in a.sizes.split(",")])


if __name__ == "__main__":
    main()
