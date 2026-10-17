"""LoCoMo long-conversation memory benchmark (Maharana et al., ACL 2024).

    python -m benchmarks.locomo --systems veramem,mem0 --convs conv-26
    python -m benchmarks.locomo --systems veramem,mem0 --convs all

Protocol (the one used by Mem0, Memory-R1 and AMA papers):
  * ingest every session of a conversation into the memory system, with session dates;
  * for each question in categories 1-4 (adversarial category 5 excluded), retrieve the
    top-k memories and answer with ONE shared answer prompt and model, so the only thing
    that differs between systems is what their memory returns;
  * score with token F1, BLEU-1 and an LLM judge (generous CORRECT/WRONG).
  * `veramem-full` additionally scores VeraMem's own answer() pipeline (citation lock +
    sufficiency retry ladder), reported separately.

Data: https://github.com/snap-research/locomo (CC BY-NC 4.0, downloaded at run time,
not redistributed). Results are resumable: re-running skips finished work.
"""

from __future__ import annotations

import argparse
import json
import re
import string
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .adapters import LLM_MODEL, OLLAMA, System

DATA_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
CATEGORY = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop"}

ANSWER_PROMPT = """You are an intelligent memory assistant tasked with retrieving accurate \
information from conversation memories.

# CONTEXT
You have access to memories from a conversation between {a} and {b}. Each memory has a \
date. They may be relevant to answering the question.

# INSTRUCTIONS
1. Carefully analyse all provided memories.
2. Pay special attention to the dates to determine the answer.
3. If the question asks about a specific event or fact, look for direct evidence.
4. If memories contain contradictory information, prioritise the most recent memory.
5. If the question involves relative time ("last year", "two months ago"), calculate the \
actual date from the memory date. Always answer with specific dates, months or years.
6. Do not confuse the two people.
7. The answer should be short: less than 5-6 words.

# MEMORIES
{memories}

Question: {question}
Reply with only the answer."""

JUDGE_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You \
will be given a question, a gold (ground truth) answer, and a generated answer.
The gold answer is usually concise. Be generous with your grading: as long as the generated \
answer touches on the same topic and contains the key information of the gold answer, it is \
CORRECT. For time questions, a generated answer in a different format (e.g. "May 7th" vs \
"7 May") or a relative expression that resolves to the same date or period is CORRECT.

Question: {q}
Gold answer: {gold}
Generated answer: {pred}

Return JSON {{"reason": "<one sentence>", "label": "CORRECT" | "WRONG"}}"""


# ------------------------------------------------------------------ metrics
def _norm(s: str) -> list[str]:
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the|and)\b", " ", s)
    return s.split()


def f1(pred: str, gold: str) -> float:
    p, g = _norm(pred), _norm(gold)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if not p or not g or same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def bleu1(pred: str, gold: str) -> float:
    p, g = _norm(pred), _norm(gold)
    if not p:
        return 0.0
    gc = Counter(g)
    clipped = sum(min(c, gc[w]) for w, c in Counter(p).items())
    import math

    bp = 1.0 if len(p) > len(g) else math.exp(1 - len(g) / max(len(p), 1))
    return bp * clipped / len(p)


# --------------------------------------------------------------------- data
def load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)  # noqa: S310 - fixed https URL
    return json.loads(path.read_text(encoding="utf-8"))


def sessions(conv: dict[str, Any]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    c = conv["conversation"]
    keys = sorted((k for k in c if re.fullmatch(r"session_\d+", k)), key=lambda k: int(k[8:]))
    out = []
    for k in keys:
        when = c.get(f"{k}_date_time", "")
        msgs = []
        for t in c[k]:
            text = t["text"]
            if t.get("blip_caption"):
                text += f" [shares a photo: {t['blip_caption']}]"
            msgs.append({"role": "user", "name": t["speaker"], "content": text, "timestamp": when})
        out.append((k, when, msgs))
    return out


# ------------------------------------------------------------------- runner
class Runner:
    def __init__(self, out_dir: Path, k: int):
        from veramem.providers import OllamaLLM, ResponseCache

        self.out = out_dir
        self.out.mkdir(parents=True, exist_ok=True)
        cache = ResponseCache(out_dir / "qa_cache.db")
        self.llm = OllamaLLM(LLM_MODEL, host=OLLAMA, cache=cache)
        self.k = k

    def system(self, name: str, conv_id: str) -> System:
        wd = self.out / name / conv_id
        wd.mkdir(parents=True, exist_ok=True)
        if name in ("veramem", "veramem-full"):
            from .adapters import VeraMemSystem

            return VeraMemSystem(
                str(self.out / "veramem" / conv_id), cache=str(self.out / "veramem_llm_cache.db")
            )
        if name == "mem0":
            from .adapters import Mem0System

            return Mem0System(str(wd))
        raise SystemExit(name)

    def ingest(self, name: str, conv: dict[str, Any], sys_: System) -> float:
        marker = (
            self.out
            / ("veramem" if name.startswith("veramem") else name)
            / conv["sample_id"]
            / "ingested.json"
        )
        done = set(json.loads(marker.read_text(encoding="utf-8"))) if marker.exists() else set()
        t = time.time()
        for key, _when, msgs in sessions(conv):
            if key in done:
                continue
            sys_.add(conv["sample_id"], msgs, session_id=key)
            done.add(key)
            marker.write_text(json.dumps(sorted(done)), encoding="utf-8")
            print(
                f"  [{name}] {conv['sample_id']} {key} ingested ({len(msgs)} turns, "
                f"{time.time() - t:.0f}s)",
                flush=True,
            )
        return time.time() - t

    def answer(self, conv: dict[str, Any], question: str, hits: list[dict[str, Any]]) -> str:
        a, b = conv["conversation"]["speaker_a"], conv["conversation"]["speaker_b"]
        mems = "\n".join(f"- {h.get('date', '')}: {h['text']}" for h in hits) or "(none)"
        out = self.llm.complete(
            [
                {
                    "role": "user",
                    "content": ANSWER_PROMPT.format(a=a, b=b, memories=mems, question=question),
                }
            ],
            max_tokens=60,
        )
        return out.strip().split("\n")[0].removeprefix("Answer:").strip()

    def judge(self, q: str, gold: str, pred: str) -> int:
        try:
            d = self.llm.json(
                [{"role": "user", "content": JUDGE_PROMPT.format(q=q, gold=gold, pred=pred)}],
                max_tokens=120,
            )
            return int(str(d.get("label", "")).upper().startswith("CORRECT"))
        except Exception:  # noqa: BLE001
            return 0

    def run(self, name: str, conv: dict[str, Any]) -> None:
        cid = conv["sample_id"]
        res_path = self.out / f"{name}.{cid}.jsonl"
        done = set()
        if res_path.exists():
            done = {
                json.loads(x)["qid"] for x in res_path.read_text(encoding="utf-8").splitlines() if x
            }
        qas = [(i, q) for i, q in enumerate(conv["qa"]) if q.get("category") in CATEGORY]
        if len(done) >= len(qas):
            print(f"[{name}] {cid}: already complete", flush=True)
            return
        sys_ = self.system(name, cid)
        ing = self.ingest(name, conv, sys_)
        today = sessions(conv)[-1][1]
        with res_path.open("a", encoding="utf-8") as f:
            for i, qa in qas:
                if i in done:
                    continue
                gold = str(qa.get("answer", ""))
                t = time.time()
                if name == "veramem-full":
                    a = sys_.mem.answer(qa["question"], user_id=cid, limit=self.k, today=today)
                    pred, ctx_n = a.answer, len(a.used_memories)
                else:
                    hits = sys_.search(cid, qa["question"], k=self.k)
                    pred, ctx_n = self.answer(conv, qa["question"], hits), len(hits)
                lat = time.time() - t
                row = {
                    "qid": i,
                    "category": CATEGORY[qa["category"]],
                    "question": qa["question"],
                    "gold": gold,
                    "pred": pred,
                    "f1": f1(pred, gold),
                    "bleu1": bleu1(pred, gold),
                    "judge": self.judge(qa["question"], gold, pred),
                    "latency_s": round(lat, 2),
                    "context_items": ctx_n,
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
        print(f"[{name}] {cid}: ingest {ing:.0f}s, qa done", flush=True)
        sys_.close()


def summarise(out: Path) -> dict[str, Any]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in out.glob("*.jsonl"):
        rows[p.name.split(".")[0]] += [
            json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x
        ]
    summary: dict[str, Any] = {}
    for name, rs in rows.items():
        by = defaultdict(list)
        for r in rs:
            by[r["category"]].append(r)
            by["overall"].append(r)
        summary[name] = {
            cat: {
                "n": len(v),
                "judge": round(100 * sum(r["judge"] for r in v) / len(v), 2),
                "f1": round(100 * sum(r["f1"] for r in v) / len(v), 2),
                "bleu1": round(100 * sum(r["bleu1"] for r in v) / len(v), 2),
            }
            for cat, v in by.items()
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="veramem,mem0")
    ap.add_argument("--convs", default="conv-26", help="comma list of sample ids, or 'all'")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--data", default="benchmarks/data/locomo10.json")
    ap.add_argument("--out", default="benchmarks/runs/locomo")
    a = ap.parse_args(argv)
    data = load(Path(a.data))
    convs = data if a.convs == "all" else [c for c in data if c["sample_id"] in a.convs.split(",")]
    r = Runner(Path(a.out), a.k)
    for conv in convs:
        for name in a.systems.split(","):
            r.run(name, conv)
            print(json.dumps(summarise(Path(a.out)), indent=1), flush=True)


if __name__ == "__main__":
    main()
