"""The two tools the agent may call, and the allow-listed dispatcher that runs them.

Everything the model sends here is untrusted: the tool name must be on the allow-list, the
arguments must be valid JSON that passes a Pydantic schema, the SQL goes through GuardedSQL.
dispatch() ALWAYS returns a string - a failure becomes an observation the model can read and
recover from, never an exception that kills the loop (Lecture 10).

Everything a tool returns is untrusted too: it is fenced as data (Lecture 14), and the system
prompt repeats that tool results can never give instructions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from pydantic import ValidationError

from app.chunking import Chunk
from app.config import Settings
from app.kevdb import SCHEMA_DOC
from app.retrieval import Retriever
from app.schemas import SearchArgs, SqlArgs
from app.sqlguard import GuardedSQL, SqlRejected

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Search NIST security guidance documents (incident response, patch management, "
                "risk assessment, log management, zero trust, malware handling, configuration "
                "management, secure software development, container security, threat information "
                "sharing, the Cybersecurity Framework). Use it for explanations, recommendations "
                "and 'what does the guidance say' questions. Write the query in English. Returns "
                "numbered passages [n] with document and page."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "English search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_kev",
            "description": (
                "Run ONE read-only SQLite SELECT over the CISA Known Exploited Vulnerabilities table. "
                "Use it for counting, filtering, ranking, dates, vendors, products, CWEs and ransomware "
                "use - never count or estimate yourself.\n" + SCHEMA_DOC
            ),
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string", "description": "a single SELECT statement"}},
                "required": ["sql"],
            },
        },
    },
]
TOOL_NAMES = frozenset(t["function"]["name"] for t in TOOL_SPECS)

PASSAGE_CHARS = 900


@dataclass
class ToolContext:
    """Per-question state: what the tools may do, and a record of what they actually did."""

    retriever: Retriever
    sql: GuardedSQL
    settings: Settings
    manifest: dict[str, dict]
    enabled: frozenset[str] = TOOL_NAMES
    sources: dict[str, dict] = field(default_factory=dict)  # chunk.key -> citation record
    sql_log: list[dict] = field(default_factory=list)

    def source_by_n(self, n: int) -> dict | None:
        return next((s for s in self.sources.values() if s["n"] == n), None)

    def _register(self, chunk: Chunk, score: float) -> int:
        """A stable [n] per chunk, recorded by OUR code - the model cannot invent a source."""
        if chunk.key not in self.sources:
            meta = self.manifest.get(chunk.doc, {})
            self.sources[chunk.key] = {
                "n": len(self.sources) + 1, "doc": chunk.doc, "page": chunk.page,
                "title": meta.get("id", chunk.doc), "score": round(score, 3),
                "text": chunk.text,  # kept for the evaluation (did the evidence reach the model?)
            }
        return self.sources[chunk.key]["n"]


def _fence(tool: str, body: str) -> str:
    return f'<tool_result tool="{tool}" trust="untrusted-data">\n{body}\n</tool_result>'


def _search_docs(ctx: ToolContext, args: SearchArgs) -> str:
    s = ctx.settings
    hits = ctx.retriever.search(args.query, mode=s.retrieval, k=s.top_k, min_score=s.min_score)
    if not hits:
        return _fence("search_docs", "NO_RELEVANT_PASSAGES: nothing in the document set scored above the relevance floor.")
    lines = []
    for h in hits:
        n = ctx._register(h.chunk, h.dense if h.dense is not None else h.score)
        title = ctx.sources[h.chunk.key]["title"]
        lines.append(f"[{n}] ({title}, p.{h.chunk.page}) {h.chunk.text[:PASSAGE_CHARS]}")
    return _fence("search_docs", "\n".join(lines))


def _query_kev(ctx: ToolContext, args: SqlArgs) -> str:
    try:
        result = ctx.sql.run(args.sql)
    except SqlRejected as exc:
        ctx.sql_log.append({"sql": args.sql, "ok": False, "n_rows": None, "error": str(exc)})
        return f"error: {exc}"
    ctx.sql_log.append({"sql": args.sql, "ok": True, "n_rows": len(result.rows), "error": None,
                        "columns": result.columns, "rows": result.rows})
    return _fence("query_kev", result.to_text())


def dispatch(ctx: ToolContext, name: str, raw_args: str | None) -> str:
    """Run ONE tool call. Never raises."""
    if name not in TOOL_NAMES or name not in ctx.enabled:
        return f"error: unknown tool {name!r}. Available: {sorted(ctx.enabled)}"
    try:
        payload = json.loads(raw_args or "{}")  # `arguments` arrives as a JSON *string*
    except json.JSONDecodeError:
        return f"error: arguments were not valid JSON: {raw_args!r}"
    if not isinstance(payload, dict):
        return "error: arguments must be a JSON object"
    try:
        if name == "search_docs":
            return _search_docs(ctx, SearchArgs(**payload))
        return _query_kev(ctx, SqlArgs(**payload))
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        return f"error: invalid arguments ({problems})"
    except Exception as exc:  # noqa: BLE001 - surface as data, keep the loop alive
        return f"error: {type(exc).__name__}: {exc}"
