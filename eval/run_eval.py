"""Reproducible evaluation of the Cyber Advisor.

    uv run python -m eval.run_eval check                       # offline: is the test set itself sound?
    uv run python -m eval.run_eval retrieval --modes bm25      # offline: BM25 baseline, both chunkings
    uv run python -m eval.run_eval retrieval                   # needs key + index: bm25/dense/hybrid x chunking
    uv run python -m eval.run_eval floor                       # needs key + index: sweep the min_score floor
    uv run python -m eval.run_eval agent --modes agent,rag_only,no_tools   # needs key: end-to-end

What is measured (see eval/dataset.jsonl; every question is labelled by hand-checked gold):
  retrieval  Hit@k / MRR / doc-recall of evidence phrases in the retrieved chunks (doc + mixed questions)
  floor      balanced accuracy of "answer vs refuse" as the cosine floor moves (in-scope vs out-of-scope)
  agent      end-to-end correctness per question type, against two baselines:
               rag_only  documents but no SQL      no_tools  the bare model, no retrieval at all
Results are written to eval/results/ so the numbers in the report can be traced to a run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

from app import llm
from app.agent import run_agent
from app.chunking import build_chunks, load_docs, load_manifest, strategy_slug
from app.config import DOCS_DIR, INDEX_DIR, get_settings
from app.embcache import embed_cached
from app.index import VectorIndex
from app.kevdb import connect_kev
from app.retrieval import Retriever
from app.runtime import build_runtime
from app.sqlguard import GuardedSQL
from eval.metrics import (answer_has_all, balanced_accuracy, has_evidence, mean, norm, retrieval_scores,
                          rows_match)

HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset.jsonl"
RESULTS = HERE / "results"
TYPES = ("doc", "sql", "mixed", "oos")


# ---- helpers ----------------------------------------------------------------------------

def load_dataset(ids: str | None = None) -> list[dict]:
    items = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    if ids:
        wanted = set(ids.split(","))
        items = [i for i in items if i["id"] in wanted]
    return items


def md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pct(passed: int, n: int) -> str:
    return f"{100 * passed / n:.0f}% ({passed}/{n})" if n else "-"


def save(name: str, payload: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    payload = {"created": dt.datetime.now().isoformat(timespec="seconds"), **payload}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def split_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


# ---- check: is the test set sound? (offline) ----------------------------------------------

def cmd_check(args: argparse.Namespace) -> int:
    data = load_dataset()
    manifest = load_manifest(DOCS_DIR)
    pages = {slug: [norm(p) for p in ps] for slug, ps in load_docs(DOCS_DIR).items()}
    sql = GuardedSQL(connect_kev())
    problems: list[str] = []
    ids = [i["id"] for i in data]
    if len(ids) != len(set(ids)):
        problems.append("duplicate ids")

    for it in data:
        tag = it["id"]
        if it["type"] not in TYPES:
            problems.append(f"{tag}: unknown type {it['type']!r}")
        if it.get("expect") not in ("answer", "refuse"):
            problems.append(f"{tag}: expect must be 'answer' or 'refuse'")
        for slug in it.get("docs", []):
            if slug not in manifest:
                problems.append(f"{tag}: unknown doc {slug!r}")
        for phrase in it.get("evidence", []):
            where = [s for s in it.get("docs", []) if any(norm(phrase) in p for p in pages.get(s, []))]
            if not where:
                problems.append(f"{tag}: evidence not found in its listed docs: {phrase[:60]!r}")
        if "gold_sql" in it:
            try:
                rows = sql.run(it["gold_sql"]).rows
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{tag}: gold_sql failed: {exc}")
                continue
            if not rows:
                problems.append(f"{tag}: gold_sql returned no rows")
            blob = norm(" ".join(str(v) for r in rows for v in r))
            for needle in it.get("answer_contains", []):
                if norm(needle) not in blob:
                    problems.append(f"{tag}: answer_contains {needle!r} is not in the gold rows {rows[:2]}")
        if it["type"] in ("sql", "mixed") and "answer_contains" not in it:
            problems.append(f"{tag}: sql/mixed items need answer_contains")

    counts = {t: sum(1 for i in data if i["type"] == t) for t in TYPES}
    unreviewed = [i["id"] for i in data if not i.get("reviewed")]
    print(f"dataset: {len(data)} questions  {counts}")
    print(f"hand-reviewed: {len(data) - len(unreviewed)}/{len(data)}"
          + (f"   (still to verify: {', '.join(unreviewed)})" if unreviewed else ""))
    for p in problems:
        print("PROBLEM:", p)
    print("OK: gold SQL runs, evidence phrases exist in the listed documents." if not problems else f"{len(problems)} problem(s).")
    return 1 if problems else 0


# ---- retrieval: chunking x mode ------------------------------------------------------------

def _query_vectors(items: list[dict]) -> dict[str, list[float]]:
    vecs = embed_cached([i["question"] for i in items], batch=12)  # cached: re-runs are free
    return {i["id"]: v for i, v in zip(items, vecs)}


def cmd_retrieval(args: argparse.Namespace) -> int:
    items = [i for i in load_dataset(args.ids) if i.get("evidence")]
    docs = load_docs(DOCS_DIR)
    modes, chunkings, k = split_csv(args.modes), split_csv(args.chunking), args.k
    qvecs: dict[str, list[float]] | None = None
    rows, details = [], {}

    for chunking in chunkings:
        chunks = build_chunks(chunking, docs)
        stem = INDEX_DIR / strategy_slug(chunking)
        index = VectorIndex.load(stem) if VectorIndex.exists(stem) else None
        for mode in modes:
            if mode != "bm25" and index is None:
                print(f"skip {chunking} / {mode}: no index (run: uv run python -m app.ingest {chunking})")
                continue
            if mode != "bm25" and qvecs is None:
                qvecs = _query_vectors(items)
            retriever = Retriever(chunks, index, llm.embed)
            per_item = []
            for it in items:
                hits = retriever.search(it["question"], mode=mode, k=k, min_score=0.0,
                                        qvec=qvecs[it["id"]] if qvecs else None)
                s = retrieval_scores([(h.chunk.doc, h.chunk.text) for h in hits], it["evidence"], it.get("docs", []), k)
                per_item.append({"id": it["id"], **s})
            row = {"chunking": chunking, "mode": mode, "n_chunks": len(chunks),
                   "hit": mean([p["hit"] for p in per_item]), "mrr": mean([p["mrr"] for p in per_item]),
                   "doc_recall": mean([p["doc_recall"] for p in per_item])}
            rows.append(row)
            details[f"{chunking}|{mode}"] = per_item

    if not rows:
        print("nothing to report.")
        return 1
    print(f"\nRetrieval over {len(items)} questions with evidence (doc + mixed), k={k}, floor off\n")
    print(md_table(["chunking", "mode", "chunks", f"Hit@{k}", "MRR", f"DocRecall@{k}"],
                   [[r["chunking"], r["mode"], r["n_chunks"], f"{r['hit']:.2f}", f"{r['mrr']:.3f}", f"{r['doc_recall']:.2f}"] for r in rows]))
    best = max(rows, key=lambda r: (r["mrr"], r["hit"]))
    misses = [p["id"] for p in details[f"{best['chunking']}|{best['mode']}"] if p["hit"] == 0]
    print(f"\nbest by MRR: CHUNKING={best['chunking']}  RETRIEVAL={best['mode']}"
          + (f"   (still missing: {', '.join(misses)})" if misses else ""))
    print("path:", save(f"retrieval_k{k}", {"k": k, "rows": rows, "per_item": details, "best": best}))
    return 0


# ---- floor: sweep min_score -----------------------------------------------------------------

def cmd_floor(args: argparse.Namespace) -> int:
    items = load_dataset(args.ids)
    stem = INDEX_DIR / strategy_slug(args.chunking)
    if not VectorIndex.exists(stem):
        print(f"no index for {args.chunking!r} - run: uv run python -m app.ingest {args.chunking}")
        return 1
    index = VectorIndex.load(stem)
    qvecs = _query_vectors(items)
    top1 = {i["id"]: float(index.scores(qvecs[i["id"]]).max()) for i in items}
    pos = [top1[i["id"]] for i in items if i["type"] in ("doc", "mixed")]
    neg = [top1[i["id"]] for i in items if i["type"] == "oos"]
    sql_only = [top1[i["id"]] for i in items if i["type"] == "sql"]
    print(f"top-1 cosine  answerable-from-docs (n={len(pos)}): min {min(pos):.2f}  median {sorted(pos)[len(pos)//2]:.2f}"
          f"\n              out-of-scope        (n={len(neg)}): max {max(neg):.2f}  median {sorted(neg)[len(neg)//2]:.2f}"
          f"\n              table-only (sql)    (n={len(sql_only)}): median {sorted(sql_only)[len(sql_only)//2]:.2f}\n")

    rows = []
    for t in [round(0.20 + 0.025 * s, 3) for s in range(0, 25)]:
        tpr_hits = [x >= t for x in pos]          # in-scope question still gets passages
        tnr_hits = [x < t for x in neg]           # out-of-scope question is refused
        rows.append({"min_score": t, "answer_rate": sum(tpr_hits) / len(pos), "refuse_rate": sum(tnr_hits) / len(neg),
                     "balanced_acc": balanced_accuracy(tpr_hits, tnr_hits)})
    best = max(rows, key=lambda r: (r["balanced_acc"], r["answer_rate"]))
    base = min(rows, key=lambda r: abs(r["min_score"] - 0.40))
    shown = [r for r in rows if r is best or r is base or round(r["min_score"] * 1000) % 50 == 0]
    print(md_table(["min_score", "in-scope still answered", "out-of-scope refused", "balanced acc"],
                   [[f"{r['min_score']:.3f}" + (" <- reference default" if r is base else "") + (" <- best" if r is best else ""),
                     f"{100 * r['answer_rate']:.0f}%", f"{100 * r['refuse_rate']:.0f}%", f"{r['balanced_acc']:.2f}"] for r in shown]))
    print(f"\nbefore (0.40): balanced accuracy {base['balanced_acc']:.2f}   after ({best['min_score']}): {best['balanced_acc']:.2f}")
    print(f"recommended: MIN_SCORE={best['min_score']}   (small sample: {len(pos)} in-scope, {len(neg)} out-of-scope)")
    print("path:", save("floor", {"chunking": args.chunking, "top1": top1, "rows": rows, "best": best, "baseline": base}))
    return 0


# ---- agent: end to end -----------------------------------------------------------------------

def score_item(item: dict, final: dict | None, ctx, gold_rows: list[tuple] | None) -> dict:
    out: dict = {"id": item["id"], "type": item["type"], "question": item["question"]}
    if final is None:
        return {**out, "error": "no final event", "correct": False}
    answer = final["answer"]
    used = set(final["tools_used"])
    out.update(answer=answer, tools_used=sorted(used), refused=final["refused"], steps=final["steps"],
               stopped=final["stopped"], grounded=final["grounded"], invalid_citations=final["invalid_citations"],
               sql=[s["sql"] for s in final["sql"]])
    kind = item["type"]
    if kind == "oos":
        out["correct"] = bool(final["refused"])
        return out
    out["routing_ok"] = used == set(item["expect_tools"])
    if item.get("evidence"):
        cited_ns = {c["n"] for c in final["citations"]}
        out["retrieved"] = any(has_evidence(s["text"], item["evidence"]) for s in ctx.sources.values())
        out["cited"] = any(has_evidence(s["text"], item["evidence"]) for s in ctx.sources.values() if s["n"] in cited_ns)
    if gold_rows is not None:
        out["exec_match"] = any(e["ok"] and rows_match(e["rows"], gold_rows, bool(item.get("ordered"))) for e in ctx.sql_log)
        out["answer_ok"] = answer_has_all(answer, item["answer_contains"])
    answered = not final["refused"]
    if kind == "doc":
        out["correct"] = answered and out["retrieved"]
    elif kind == "sql":
        out["correct"] = answered and out["exec_match"] and out["answer_ok"]
    else:  # mixed: both halves
        out["correct"] = answered and out["retrieved"] and out["exec_match"] and out["answer_ok"]
    return out


def _summarise(records: list[dict]) -> list[list]:
    rows = []
    for kind in TYPES:
        rs = [r for r in records if r["type"] == kind]
        if not rs:
            continue
        n = len(rs)
        def c(key):  # count of records with a truthy key, among those that have it
            have = [r for r in rs if key in r]
            return pct(sum(bool(r[key]) for r in have), len(have)) if have else "-"
        rows.append([kind, n, c("correct"), c("exec_match"), c("answer_ok"), c("retrieved"), c("cited"), c("routing_ok")])
    in_scope = [r for r in records if r["type"] != "oos"]
    oos = [r for r in records if r["type"] == "oos"]
    rows.append(["**in-scope overall**", len(in_scope), pct(sum(r["correct"] for r in in_scope), len(in_scope)), "", "", "", "", ""])
    if oos:
        rows.append(["**out-of-scope refused**", len(oos), pct(sum(r["correct"] for r in oos), len(oos)), "", "", "", "", ""])
    false_ref = sum(1 for r in in_scope if r.get("refused"))
    rows.append(["false refusals (in-scope)", len(in_scope), pct(false_ref, len(in_scope)), "", "", "", "", ""])
    return rows


def cmd_agent(args: argparse.Namespace) -> int:
    runtime = build_runtime(get_settings())
    items = load_dataset(args.ids)
    gold = {i["id"]: runtime.sql.run(i["gold_sql"]).rows for i in items if "gold_sql" in i}
    s = runtime.settings
    print(f"config: model={s.chat_model} chunking={s.chunking} retrieval={s.retrieval} top_k={s.top_k} "
          f"min_score={s.min_score} max_steps={s.max_steps}\n")
    all_records: dict[str, list[dict]] = {}
    for mode in split_csv(args.modes):
        records = []
        for it in items:
            ctx, final, started = runtime.new_context(), None, time.time()
            try:
                for ev in run_agent(it["question"], ctx, mode=mode, snapshot=runtime.snapshot):
                    if ev["type"] == "final":
                        final = ev
            except Exception as exc:  # noqa: BLE001 - record the failure, keep evaluating
                records.append({"id": it["id"], "type": it["type"], "question": it["question"],
                                "error": f"{type(exc).__name__}: {exc}", "correct": False})
                print(f"  [{mode}] {it['id']}: ERROR {type(exc).__name__}: {exc}")
                continue
            rec = score_item(it, final, ctx, gold.get(it["id"]))
            rec["seconds"] = round(time.time() - started, 1)
            records.append(rec)
            print(f"  [{mode}] {it['id']}: {'ok ' if rec['correct'] else 'MISS'} tools={rec.get('tools_used')} steps={rec.get('steps')}")
        all_records[mode] = records
        print(f"\n### mode = {mode}\n")
        print(md_table(["type", "n", "correct", "SQL exec-match", "answer has value", "evidence retrieved", "evidence cited", "right tools"],
                       _summarise(records)))
        print()
    print("path:", save("agent", {"config": s.__dict__, "snapshot": runtime.snapshot, "records": all_records}))
    return 0


# ---- entry point -----------------------------------------------------------------------------

def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to a legacy code page
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="validate the test set (offline)").set_defaults(fn=cmd_check)

    r = sub.add_parser("retrieval", help="Hit@k / MRR per chunking x retrieval mode")
    r.add_argument("--chunking", default="naive:800:150,section:1200")
    r.add_argument("--modes", default="bm25,dense,hybrid")
    r.add_argument("--k", type=int, default=5)
    r.add_argument("--ids", help="comma-separated question ids")
    r.set_defaults(fn=cmd_retrieval)

    f = sub.add_parser("floor", help="sweep the min_score floor")
    f.add_argument("--chunking", default=get_settings().chunking)
    f.add_argument("--ids")
    f.set_defaults(fn=cmd_floor)

    a = sub.add_parser("agent", help="end-to-end run against the baselines")
    a.add_argument("--modes", default="agent,rag_only,no_tools")
    a.add_argument("--ids")
    a.set_defaults(fn=cmd_agent)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
