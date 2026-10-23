# Cyber Advisor

An LLM assistant that answers security questions **from sources it can cite** and from a **table it can query exactly**:

* **Documents** - 12 NIST publications (incident response, patch management, risk, logging, zero trust, malware, container security, ...), searched with embeddings / BM25 and cited as `[n]` with document and page.
* **A table** - CISA's Known Exploited Vulnerabilities (KEV) catalog, 1 754 rows. Counting, ranking and date questions go to **guarded text-to-SQL**, because a vector search over rows cannot count.

A small tool-calling agent decides which source a question needs (or both, or neither) and declines what the sources cannot answer.

> Built for the KMITL *Building LLM-Powered Applications* capstone. It is **not** the course reference app (`kmitl-llm-app-demo`): different domain, a table + SQL tool, a hand-built 36-question evaluation with baselines, and a chunk-level retrieval metric. See [How this differs from the reference app](#how-this-differs-from-the-reference-app).

```
 browser (web/)  --POST /api/ask-->  FastAPI (app/main.py)  --SSE: act / obs / final-->  browser
                                         |
                                  agent loop (app/agent.py)   native tool calling, max_steps, duplicate guard
                                   /                      \
                    search_docs(query)                 query_kev(sql)
              retrieval.py: dense | bm25 | hybrid        sqlguard.py: SELECT-only, allow-listed
              + min_score floor + citations              table, row cap, timeout
                       |                                          |
        data/docs/*.txt  (NIST, public domain)      data/raw/known_exploited_vulnerabilities.csv (CC0)
```

## Run it

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). The data is committed, so no download is needed.

```bash
uv sync
cp .env.example .env            # then put your course key in OPENAI_API_KEY (never commit .env)
uv run python -m app.ingest     # embed the chunks once (needs the key; ~3 000 chunks)
uv run uvicorn app.main:app --port 8000     # open http://localhost:8000
uv run pytest -q                # 68 tests; need no key and no network
```

`scripts/fetch_data.py` re-downloads everything (`--force` refreshes the KEV snapshot - which changes weekly, so the committed snapshot is what the evaluation gold refers to).

## What each part is for (course lecture in brackets)

| File | Role |
|---|---|
| `app/llm.py` | the only place a model is called: `chat()` and `embed()` through the class proxy; key and base URL from the environment [L1-L4] |
| `app/schemas.py` | Pydantic models for the request, tool arguments and the final answer envelope [L2] |
| `app/main.py`, `web/` | FastAPI + Server-Sent Events backend and a no-build UI [L3-L4] |
| `app/chunking.py`, `app/index.py`, `app/retrieval.py`, `app/bm25.py` | chunking strategies, vector index, BM25, hybrid (reciprocal rank fusion), `min_score` floor [L5-L8] |
| `app/kevdb.py`, `app/sqlguard.py` | the KEV table and the guard around model-written SQL [L9, L11] |
| `app/tools.py`, `app/agent.py` | the two tools, the allow-listed dispatcher and the agent loop [L10-L11] |
| `eval/` | labelled test set, metrics, and the runner [L8] |
| `tests/` | unit + in-process API tests with a scripted fake model (no key needed) |

## Safety: where untrusted data crosses a boundary

| Boundary | What is untrusted | Control |
|---|---|---|
| browser -> API | the question | Pydantic length/shape limits (422 before any code runs) |
| model -> tools | tool name, arguments | tool-name allow-list per mode; JSON parse; Pydantic schema; `dispatch()` returns an error *string*, never raises |
| model -> database | the SQL | SQLite **authorizer** (engine-level allow-list: SELECT, table `kev`, plain functions; recursive CTEs refused), `PRAGMA query_only`, one statement, 2 000 chars, 2 s timeout, 50-row cap |
| tools -> model | document text, SQL results | fenced as `<tool_result trust="untrusted-data">`; the system prompt says results are data, never instructions |
| model -> user | citations | `[n]` markers are checked against what our code actually retrieved; an invented one is flagged, never shown as a source |
| config | the API key | read from the environment only; `/api/health` reports *whether* it is set, never the value |

The SQL guard is tested against `DROP`, `ATTACH`, `PRAGMA`, `sqlite_master`, stacked statements, recursive CTEs, and writes hidden behind a `WITH` (`tests/test_sqlguard.py`). In a server database the equivalent strongest layer is a read-only role with `SELECT` on the one allowed table.

## Evaluation

**Test set** - `eval/dataset.jsonl`, 36 questions, each with a machine-checkable gold:

| type | n | gold |
|---|---|---|
| `doc` - answerable from the NIST documents | 12 | an *evidence phrase* that must appear in a retrieved chunk |
| `sql` - answerable from the KEV table | 11 | gold SQL; judged by **execution match** (same values) and by the answer containing the value |
| `mixed` - needs both | 6 | both of the above |
| `oos` - must be declined (3 off-topic, 2 near-domain traps, 1 prompt-injection, 1 Thai) | 7 | refusal |

`uv run python -m eval.run_eval check` proves the set is sound (gold SQL runs, every evidence phrase exists in the document it names). Every item still carries `"reviewed": false` - **verify each gold by hand and flip it**, because you must be able to defend every number.

**Why a chunk-level metric.** Document-level recall cannot see chunking at all: with every chunking strategy it scores 1.00 (measured below). A window that cuts the answer sentence in half still "found the right PDF". So retrieval is scored on whether a retrieved *chunk contains the evidence phrase* (Hit@k, MRR).

**Baselines and experiments** (`uv run python -m eval.run_eval ...`):

| command | compares | needs key |
|---|---|---|
| `retrieval` | chunking (`naive:800:150` = reference default vs boundary-aware `section:N`) x mode (BM25-only / dense / hybrid) | dense, hybrid |
| `floor` | `min_score` from 0.20 to 0.80 vs the reference's 0.40: answer-rate on in-scope vs refusal-rate on out-of-scope | yes |
| `agent` | the full agent vs **documents-only** (no SQL) vs **no tools** (bare model) | yes |

**Measured so far (no key needed)** - BM25 only, k = 5, 18 questions with evidence, floor off:

| chunking | chunks | Hit@5 | MRR | DocRecall@5 |
|---|---|---|---|---|
| `naive:800:150` (default) | 2 896 | 0.67 | 0.379 | 1.00 |
| `section:600` | 4 160 | 0.61 | 0.463 | 1.00 |
| `section:1200` | 1 986 | 0.61 | 0.412 | 1.00 |

Reading it honestly: DocRecall is saturated (useless for tuning); Hit@5 is far from 1, so there is room to improve; 18 questions is small (one question = 0.056), so differences below ~0.1 are within noise. The misses are real vocabulary mismatches, not harness bugs - e.g. *"keep timestamps consistent across systems"* vs the passage about *NTP servers and clock synchronisation* is exactly what dense retrieval should fix.

**Not yet measured:** dense and hybrid retrieval, the `min_score` sweep, and the end-to-end agent comparison. They need the class proxy; run the three commands above and paste the tables into the report. The chunking and floor choices are made on the same 36 questions they are reported on, which the report must say.

## How this differs from the reference app

1. **Corpus the reference cannot answer** - cybersecurity (NIST + CISA KEV), raw data and fetch script committed (`data/`, `scripts/fetch_data.py`, `data/SOURCES.md`).
2. **Own evaluation** - 36 questions, a named baseline per family, reproducible from this repo.
3. **A measured improvement over the naive default** - chunking, retrieval mode and `min_score` are swept, not guessed.
4. **A second source type** - a table with guarded text-to-SQL, which the reference does not have; native function calling instead of the reference's text protocol.

## Limitations

* English-only evaluation. The agent is *instructed* to reply in the question's language and the embedding model is multilingual, but Thai quality is unmeasured, and the BM25 tokenizer ignores non-Latin text (hybrid mode still has the dense half).
* NIST publications are static PDFs; extraction is imperfect (tables flatten). The KEV snapshot ages every week.
* A small model (`gemma-4-E4B-it`) writes the SQL. Hard cases (exact CWE matching, percentages) are tagged `"hard": true` in the dataset so failures are visible rather than averaged away.
* cisa.gov HTML pages answer non-browser clients with HTTP 403, and this project does not work around that; CISA advisories are therefore not in the corpus.

## Data and licences

See [`data/SOURCES.md`](data/SOURCES.md). KEV: CC0 1.0. NIST publications: works of the US Government, not subject to copyright in the US. Verify each PDF's own notice before redistributing outside the US.
