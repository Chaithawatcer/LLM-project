"""Reference-guided LLM-as-judge (Lecture 8) for the document half of an answer.

Grading free text is hard; grading it *against a known sentence* is much easier: we already have,
for every document question, the sentence from the NIST text that holds the answer. The judge only
decides "does this answer state that fact?" - a YES/NO, temperature 0.

Known weaknesses, so the report can say them out loud: the judge is the same small model that wrote
the answer (self-preference), and it can be lenient with vague answers. Calibrate it: read a sample of
`fact_ok` decisions in eval/results/*.json against the answer and the reference and count agreement.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app import llm

SYSTEM = (
    "You grade an answer against one reference fact. Reply with exactly YES or NO.\n"
    "YES only if the answer states the reference fact or a clear paraphrase of it.\n"
    "NO if the answer contradicts it, leaves it out, is vague, or says the information is not available.\n"
    "Do not use outside knowledge: judge only whether the reference fact is present."
)


def judge_fact(question: str, answer: str, fact: str, *, chat_fn: Callable[..., Any] = llm.chat) -> bool:
    msg = chat_fn(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Question: {question}\nReference fact: {fact}\nAnswer: {answer}\n\nDoes the answer state the reference fact? YES or NO."},
        ],
        temperature=0.0,
        max_tokens=5,
    )
    return (getattr(msg, "content", "") or "").strip().upper().startswith("YES")
