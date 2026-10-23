"""Run model-written SQL safely (Lecture 9/11: generated SQL is untrusted input).

Layers, strongest first - the model's SQL has to get past all of them:

  1. a SQLite *authorizer*: the engine itself asks us, for every operation in the parsed
     statement, whether it may happen. We allow only SELECT, reads of the `kev` table, and a
     short list of harmless functions. INSERT/UPDATE/DELETE/DROP/ATTACH/PRAGMA, reads of
     sqlite_master, recursive CTEs (infinite loops) ... are refused by the parser, not by a
     regex that a clever string could slip past.
  2. `PRAGMA query_only = ON`: even if (1) had a hole, the connection cannot write.
  3. one statement only (sqlite3 itself refuses two), 2 000 characters at most.
  4. a wall-clock timeout (progress handler) and a row cap, so a bad query cannot hang or flood us.

Production note: in a real database the strongest layer is a read-only *role* with
SELECT granted on the allow-listed table only - same idea, enforced by the server.
"""
from __future__ import annotations

import re
import sqlite3
import threading
import time
from dataclasses import dataclass

ALLOWED_TABLES = frozenset({"kev"})
ALLOWED_FUNCTIONS = frozenset(
    """count sum avg min max total lower upper substr substring length strftime date datetime julianday
    coalesce ifnull nullif iif round abs instr trim ltrim rtrim replace like glob group_concat printf
    row_number rank dense_rank""".split()
)
MAX_SQL_CHARS = 2000
MAX_ROWS = 50
TIMEOUT_S = 2.0

_SQLITE_RECURSIVE = getattr(sqlite3, "SQLITE_RECURSIVE", 33)


class SqlRejected(ValueError):
    """The query was refused (not a single read-only SELECT over the allow-listed table)."""


def _authorizer(action: int, arg1, arg2, dbname, source) -> int:
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:  # arg1 = table, arg2 = column
        return sqlite3.SQLITE_OK if (arg1 or "").lower() in ALLOWED_TABLES else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:  # arg2 = function name
        return sqlite3.SQLITE_OK if (arg2 or "").lower() in ALLOWED_FUNCTIONS else sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_DENY  # everything else, including SQLITE_RECURSIVE


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool

    def to_text(self, max_cell: int = 160) -> str:
        def cell(v) -> str:
            s = "" if v is None else str(v)
            return s if len(s) <= max_cell else s[: max_cell - 1] + "…"

        lines = [f"columns: {' | '.join(self.columns)}", f"rows ({len(self.rows)}{'+' if self.truncated else ''}):"]
        lines += [" | ".join(cell(v) for v in row) for row in self.rows]
        if self.truncated:
            lines.append(f"(truncated at {MAX_ROWS} rows - add a WHERE/LIMIT or aggregate)")
        return "\n".join(lines)


class GuardedSQL:
    """Wraps a connection so that it can only run guarded SELECTs. Thread-safe."""

    def __init__(self, con: sqlite3.Connection) -> None:
        con.execute("PRAGMA query_only = ON")  # must come before the authorizer denies PRAGMA
        con.set_authorizer(_authorizer)
        self._con = con
        self._lock = threading.Lock()

    def run(self, sql: str, *, max_rows: int = MAX_ROWS, timeout_s: float = TIMEOUT_S) -> QueryResult:
        sql = (sql or "").strip().rstrip(";").strip()
        if not sql:
            raise SqlRejected("empty query")
        if len(sql) > MAX_SQL_CHARS:
            raise SqlRejected(f"query longer than {MAX_SQL_CHARS} characters")
        if not re.match(r"(?is)^(select|with)\b", sql):
            raise SqlRejected("only a single SELECT (optionally starting with WITH) is allowed")
        deadline = time.monotonic() + timeout_s
        with self._lock:
            self._con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
            try:
                cur = self._con.execute(sql)
                columns = [d[0] for d in (cur.description or [])]
                rows = cur.fetchmany(max_rows + 1)
            except sqlite3.DatabaseError as exc:  # syntax error, denied operation, timeout, 2 statements ...
                raise SqlRejected(_explain(exc)) from exc
            except sqlite3.Warning as exc:  # older Pythons raise Warning for "one statement at a time"
                raise SqlRejected(str(exc)) from exc
            finally:
                self._con.set_progress_handler(None, 0)
        return QueryResult(columns, [tuple(r) for r in rows[:max_rows]], truncated=len(rows) > max_rows)


def _explain(exc: Exception) -> str:
    msg = str(exc)
    if "not authorized" in msg:
        return "not authorized: only SELECT over the `kev` table with plain functions is allowed"
    if "interrupted" in msg:
        return f"query took longer than {TIMEOUT_S:.0f}s and was stopped"
    return msg
