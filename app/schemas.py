"""Request/response shapes. Pydantic validates everything that crosses a boundary:
the browser's request, the model's tool arguments, and the final answer we hand back."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Mode = Literal["agent", "rag_only", "no_tools"]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    # "agent"    = both tools (the app)
    # "rag_only" = document search only   (baseline for table questions)
    # "no_tools" = the bare model         (baseline: no retrieval at all)
    mode: Mode = "agent"


class SearchArgs(BaseModel):
    """Arguments the model may pass to search_docs."""
    query: str = Field(..., min_length=2, max_length=300)


class SqlArgs(BaseModel):
    """Arguments the model may pass to query_kev."""
    sql: str = Field(..., min_length=6, max_length=2000)


class Citation(BaseModel):
    n: int
    doc: str
    title: str
    page: int
    score: float | None = None


class SqlRun(BaseModel):
    sql: str
    ok: bool
    n_rows: int | None = None
    error: str | None = None


class FinalAnswer(BaseModel):
    answer: str
    citations: list[Citation]          # only the [n] markers the answer actually uses
    invalid_citations: list[int]       # [n] markers that point at nothing we retrieved
    sql: list[SqlRun]
    tools_used: list[str]
    grounded: bool                     # False = the model answered without reading any tool result
    refused: bool                      # True = it declined with the fixed refusal sentence
    steps: int
    stopped: Literal["answered", "max_steps"]
