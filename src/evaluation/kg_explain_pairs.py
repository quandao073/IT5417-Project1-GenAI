"""Layer 4, the value of the knowledge part (session D): on the same search results, the
explanation with knowledge (prompt v4) against the one without (v3), compared by Opus blind and in
a seeded random order (prompt `kg_explain_judge_v1`), on two criteria judged apart:
(a) faithfulness to the data the writer was given, (b) usefulness to the user.
"""

from __future__ import annotations

import json
import random

import numpy as np

CRITERIA = ("faithfulness", "usefulness")
_VERDICT = {"type": "object", "additionalProperties": False, "required": ["winner", "reason"],
            "properties": {"winner": {"type": "string", "enum": ["A", "B", "tie"]},
                           "reason": {"type": "string"}}}
SCHEMA = {"type": "object", "additionalProperties": False, "required": list(CRITERIA),
          "properties": {c: _VERDICT for c in CRITERIA}}


def render(event: dict) -> str:
    """An explanation as the user reads it: summary, one line per image, the knowledge part."""
    lines = [event.get("summary") or ""]
    lines += [f"#{i['rank']}: {i['line']}" for i in event.get("images") or []]
    if event.get("knowledge"):
        lines += ["", event["knowledge"]]
    return "\n".join(lines).strip()


def swapped(qid: str, seed: int) -> bool:
    """True: the explanation with knowledge is B. Seeded per question."""
    return random.Random(f"{seed}:{qid}").random() < 0.5


def user_message(facts: dict, a: str, b: str) -> str:
    return (f"DATA:\n{json.dumps(facts, ensure_ascii=False, indent=1)}\n\n"
            f"EXPLANATION A:\n{a}\n\nEXPLANATION B:\n{b}")


def outcome(answer: dict, criterion: str, knowledge_is_b: bool) -> int:
    """+1 the explanation with knowledge wins, -1 it loses, 0 tie."""
    w = answer[criterion]["winner"]
    if w == "tie":
        return 0
    return 1 if (w == "B") == knowledge_is_b else -1


def tally(outcomes: list[int], n_boot: int, seed: int) -> dict:
    """Wins / losses / ties of the explanation with knowledge, the mean of +1/0/-1 and the
    win share (ties half) with 95% bootstrap CIs over questions."""
    x = np.asarray(outcomes, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"n": len(x), "wins": int((x > 0).sum()), "losses": int((x < 0).sum()),
            "ties": int((x == 0).sum()), "mean": float(x.mean()), "ci95": [float(lo), float(hi)],
            "win_share": float((x.mean() + 1) / 2),
            "win_share_ci95": [float((lo + 1) / 2), float((hi + 1) / 2)]}
