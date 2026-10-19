"""Citation-locked answering.

1. Retrieve (routed, hybrid).
2. One LLM call selects supporting memories, answers, and judges sufficiency
   (Memory-R1 answer-agent distillation + AMA judge, fused to save a call).
3. Citation lock (Agent Zero Memory, Def. 2): only ids the model was actually shown
   may be cited; anything else is stripped.
4. If evidence is insufficient, walk AMA's retry ladder: dual channel -> source turns of
   the retrieved facts -> neighbouring turns. Abstain if still insufficient.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any

from . import prompts
from .policy import Principal
from .providers.llm import LLMError
from .types import Answer, Hit, day, new_id, now

if TYPE_CHECKING:
    from .memory import Memory

_IDK = re.compile(
    r"^\s*(i don'?t know|unknown|not (enough|sufficient) information|"
    r"no information|cannot be determined)",
    re.I,
)


def render(hits: list[Hit]) -> tuple[str, dict[str, Hit]]:
    lines, labels = [], {}
    for i, h in enumerate(hits, 1):
        m = h.memory
        when = day(m.valid_from or m.created_at)
        tag = "shared experience, auxiliary" if h.granularity == "shared" else m.authority.value
        kind = "raw message" if h.granularity == "raw" else m.kind.value
        lines.append(f"[{i}] ({when}; {kind}; {tag}) {m.text}")
        labels[str(i)] = h
    return "\n".join(lines), labels


class Answerer:
    def __init__(self, mem: Memory):
        self.m = mem

    def answer(
        self,
        question: str,
        p: Principal,
        limit: int = 20,
        style: str = "concise",
        max_rounds: int = 2,
        now_hint: str | None = None,
    ) -> Answer:
        r = self.m.retriever
        res = r.search(question, p, limit=limit, record=False)
        hits = list(res.hits)
        shown: dict[str, Hit] = {}
        out: dict[str, Any] = {"answer": "I don't know", "selected": [], "sufficient": False}
        for rnd in range(max_rounds + 1):
            ctx, labels = render(hits)
            shown = labels
            out = self._ask(question, ctx, style, now_hint)
            if out.get("sufficient", True) and not _IDK.match(str(out.get("answer", ""))):
                break
            if rnd == max_rounds:
                break
            # retry ladder
            if rnd == 0 and res.granularity not in ("dual", "raw"):
                extra = r.search(question, p, limit=limit, granularity="dual", record=False).hits
            else:
                extra = r.source_episodes(hits[:limit], p, neighbours=1)
            known = {h.memory.id for h in hits}
            new = [h for h in extra if h.memory.id not in known]
            if not new:
                break
            hits = hits + new[:limit]

        cited = []
        selected = out.get("selected")
        for c in selected if isinstance(selected, list) else []:
            key = str(c).strip("[] ")
            if key in shown:  # citation lock
                cited.append(shown[key].memory.id)
        text = str(out.get("answer", "")).strip() or "I don't know"
        abstained = bool(_IDK.match(text)) or (not out.get("sufficient", True) and not cited)
        if abstained:
            text = "I don't know"
        trace_id = new_id("tr")
        used = cited or [h.memory.id for h in hits[:5]]
        self.m.store.put_trace(
            trace_id,
            "answer",
            question,
            p.user_id,
            p.agent_id,
            now(),
            [h.memory.id for h in hits],
            {"cited": cited, "answer": text, "abstained": abstained, "rounds": rnd + 1},
        )
        self.m.retriever._touch([i for i in cited if i.startswith("mem_")])
        return Answer(
            trace_id=trace_id, answer=text, citations=cited, abstained=abstained, used_memories=used
        )

    def _ask(self, question: str, ctx: str, style: str, now_hint: str | None) -> dict:
        system = prompts.ANSWER_SYSTEM.replace("{style}", prompts.ANSWER_STYLE.get(style, ""))
        today = now_hint or time.strftime("%Y-%m-%d")
        user = f"Today is {today}.\n\nMEMORIES:\n{ctx or '(none)'}\n\nQUESTION: {question}"
        try:
            d = self.m.llm.json(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=600,
            )
        except LLMError as e:
            return {"answer": "I don't know", "selected": [], "sufficient": False, "error": str(e)}
        return d if isinstance(d, dict) else {"answer": str(d), "selected": [], "sufficient": True}
