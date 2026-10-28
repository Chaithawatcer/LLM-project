"""Build the Thai project report (PDF) from report.template.html and the evaluation results.

    uv run python docs/report/build_report.py

Every table and chart is computed from eval/results/*.json and eval/dataset.jsonl, and the numbers quoted in the
prose are asserted against them (EXPECTED below), so the report cannot drift away from the data: if a result file
changes, the build stops and tells you which sentence to update.

Fill in the four constants below, then re-run. The PDF is rendered by headless Edge/Chrome (they shape Thai text
correctly; ReportLab does not).
"""
from __future__ import annotations

import datetime as dt
import html
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---- fill these in --------------------------------------------------------------------------------
NAME = "(ใส่ชื่อ-นามสกุล)"
STUDENT_ID = "(ใส่รหัสนักศึกษา)"
GITHUB_URL = "(ใส่ลิงก์ GitHub ของ repo)"
LIVE_URL = "(ใส่ลิงก์ระบบที่ deploy แล้ว)"
# ----------------------------------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RESULTS = ROOT / "eval" / "results"
OUT_HTML = HERE / "report.html"
OUT_PDF = HERE / "Cyber-Advisor-Report-TH.pdf"
THAI_MONTHS = ["มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน", "กรกฎาคม", "สิงหาคม",
               "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม"]


def load_records(tag: str, mode: str = "agent") -> list[dict]:
    return json.loads((RESULTS / f"agent_{tag}.json").read_text(encoding="utf-8"))["records"][mode]


def load_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def frac(num: int, den: int) -> str:
    return f"{num}/{den}"


def pct(num: int, den: int) -> str:
    return f"{100 * num / den:.0f}%" if den else "-"


def summarise(records: list[dict]) -> dict:
    ins = [r for r in records if r["type"] != "oos"]
    oos = [r for r in records if r["type"] == "oos"]

    def have(kind, key):
        rs = [r for r in records if r["type"] == kind and key in r]
        return sum(bool(r[key]) for r in rs), len(rs)

    def by_type(kind):
        rs = [r for r in records if r["type"] == kind]
        return sum(r["correct"] for r in rs), len(rs)

    return {
        "n_in": len(ins), "correct": sum(r["correct"] for r in ins),
        "verified": sum(bool(r.get("verified")) for r in ins),
        "doc_retr": have("doc", "retrieved"), "mixed_retr": have("mixed", "retrieved"),
        "sql_exec": have("sql", "exec_match"),
        "oos_ok": sum(r["correct"] for r in oos), "n_oos": len(oos),
        "false_ref": sum(1 for r in ins if r.get("refused")),
        "type": {k: by_type(k) for k in ("doc", "sql", "mixed", "oos")},
    }


# ---- load everything ---------------------------------------------------------------------------------
dataset = [json.loads(line) for line in (ROOT / "eval" / "dataset.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
retrieval = load_json("retrieval.json")
floor = load_json("floor.json")
RUNS = {  # key: (label, config, records)
    "A": ("A  ค่าตั้งต้นของแอปตัวอย่าง", "dense, k=5, floor 0.40", load_records("A_baseline")),
    "B": ("B  ปรับแล้ว (ค่าเริ่มต้น)", "hybrid, k=10, floor 0.55", load_records("B_hybrid_k10_floor055")),
    "B2": ("B2 รัน B ซ้ำ", "เหมือน B", load_records("B2_replicate")),
    "C": ("C  prompt v2", "B + กฎ routing", load_records("C_B_plus_prompt_v2")),
    "C2": ("C2 รัน C ซ้ำ", "เหมือน C", load_records("C2_replicate")),
    "RAG": ("Baseline: เอกสารอย่างเดียว", "B ไม่มีเครื่องมือ SQL", load_records("D_baselines", "rag_only")),
    "BARE": ("Baseline: โมเดลเปล่า", "ไม่มีเครื่องมือเลย", load_records("D_baselines", "no_tools")),
}
S = {k: summarise(v[2]) for k, v in RUNS.items()}

# ---- the numbers the prose quotes; the build fails if the data disagrees ------------------------------
EXPECTED = {
    ("A", "correct"): 19, ("A", "verified"): 16, ("B", "correct"): 22, ("B", "verified"): 20,
    ("B2", "correct"): 23, ("B2", "verified"): 21, ("C", "correct"): 23, ("C2", "correct"): 23,
    ("C", "verified"): 22, ("C2", "verified"): 22, ("RAG", "correct"): 9, ("BARE", "correct"): 5,
    ("A", "n_in"): 29, ("BARE", "oos_ok"): 0, ("RAG", "oos_ok"): 6, ("B2", "oos_ok"): 7, ("A", "oos_ok"): 7,
}
for (run, key), want in EXPECTED.items():
    assert S[run][key] == want, f"report text assumes {run}.{key} = {want}, data says {S[run][key]}"
assert S["RAG"]["type"]["sql"][0] == 0 and S["RAG"]["type"]["mixed"][0] == 0, "prose says documents-only gets 0/17"
assert S["B2"]["type"]["sql"] == (10, 11) and S["B2"]["type"]["doc"] == (9, 12) and S["B2"]["type"]["mixed"] == (4, 6)
assert S["A"]["doc_retr"] == (5, 12) and S["B"]["doc_retr"] == (8, 12) and S["A"]["mixed_retr"] == (3, 6) and S["B"]["mixed_retr"] == (5, 6)
for r in ("B", "B2"):
    assert next(x for x in RUNS[r][2] if x["id"] == "s11")["correct"] is False
flipped = [a["id"] for a, b in zip(RUNS["B"][2], RUNS["B2"][2]) if a["correct"] != b["correct"]]
assert flipped == ["m02"], f"prose says B and B2 differ only on m02, data says {flipped}"


def t(rows, header, cls=""):
    head = "".join(f"<th>{h}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


# ---- tables and chart ----------------------------------------------------------------------------------
KIND_TH = {"doc": "ถามจากเอกสาร (doc)", "sql": "ถามจากตาราง (sql)", "mixed": "ถามผสมสองแหล่ง (mixed)", "oos": "ต้องปฏิเสธ (oos)"}
GOLD_TH = {
    "doc": "วลีหลักฐานที่ต้องอยู่ใน chunk ที่ดึงมา; ข้อเท็จจริงอ้างอิงสำหรับตัวตัดสิน",
    "sql": "SQL เฉลย (เทียบผลลัพธ์) และค่าที่คำตอบต้องมี",
    "mixed": "ทั้งสองอย่างข้างต้น",
    "oos": "ต้องปฏิเสธ (นอกเรื่อง 3, ใกล้โดเมนแต่ตอบไม่ได้ 2, โจมตี prompt 1, ภาษาไทย 1)",
}
dataset_table = t([[KIND_TH[k], sum(1 for d in dataset if d["type"] == k), GOLD_TH[k]] for k in ("doc", "sql", "mixed", "oos")],
                  ["ประเภท", "ข้อ", "เฉลย / วิธีตรวจ"])
n_reviewed = sum(1 for d in dataset if d.get("reviewed"))
n_total = len(dataset)
review_note = ("เฉลยทุกข้อได้รับการตรวจทานด้วยมือโดยผู้จัดทำ" if n_reviewed == n_total else
               f"เฉลยผ่านการตรวจโดยอัตโนมัติ (SQL เฉลยรันได้ทุกข้อ และวลีหลักฐานทุกข้อมีอยู่ในเอกสารที่ระบุ); "
               f"ณ วันที่สร้างรายงาน {n_reviewed} จาก {n_total} ข้อผ่านการตรวจทานด้วยมือ")

ks = retrieval["ks"]
MODE_TH = {"bm25": "BM25 อย่างเดียว", "dense": "dense (ค่าตั้งต้นของแอปตัวอย่าง)", "hybrid": "<b>hybrid (RRF)</b>"}
retrieval_table = t(
    [[r["chunking"].replace(":", " : "), MODE_TH[r["mode"]], r["n_chunks"], *[f"{r[f'hit@{k}']:.2f}" for k in ks],
      f"{r['mrr']:.3f}", f"{r['doc_recall']:.2f}"] for r in retrieval["rows"]],
    ["chunking", "วิธีค้น", "chunk", *[f"Hit@{k}" for k in ks], f"MRR@{ks[0]}", f"DocRecall@{ks[0]}"], "num")
hy = next(r for r in retrieval["rows"] if r["chunking"] == "naive:800:150" and r["mode"] == "hybrid")
de = next(r for r in retrieval["rows"] if r["chunking"] == "naive:800:150" and r["mode"] == "dense")
assert (round(hy["hit@20"], 2), round(de["hit@20"], 2), round(hy["hit@5"], 2)) == (0.94, 0.67, 0.67)

types = {d["id"]: d["type"] for d in dataset}
in_scope_top1 = [v for k, v in floor["top1"].items() if types[k] in ("doc", "mixed")]
oos_top1 = [v for k, v in floor["top1"].items() if types[k] == "oos"]
wanted = {0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70}
base_r, best_r = floor["baseline"], floor["best"]
floor_table = t(
    [[f"{r['min_score']:.2f}" + (" (ค่าตั้งต้น)" if r["min_score"] == base_r["min_score"] else "") + (" <b>(ที่เลือก)</b>" if r["min_score"] == best_r["min_score"] else ""),
      f"{100 * r['answer_rate']:.0f}%", f"{100 * r['refuse_rate']:.0f}%", f"{r['balanced_acc']:.2f}"]
     for r in floor["rows"] if round(r["min_score"], 2) in wanted],
    ["min_score", "ข้อที่ตอบได้ ยังตอบ", "ข้อนอกขอบเขต ถูกปฏิเสธ", "balanced accuracy"], "num2")
assert (base_r["min_score"], best_r["min_score"]) == (0.4, 0.55)
assert round(base_r["refuse_rate"], 2) == 0.29 and round(best_r["refuse_rate"], 2) == 1.0 and round(best_r["answer_rate"], 2) == 0.94


def cell(k):
    s = S[k]
    return [RUNS[k][0], RUNS[k][1], f"{s['correct']} ({pct(s['correct'], s['n_in'])})", f"{s['verified']} ({pct(s['verified'], s['n_in'])})",
            frac(*s["doc_retr"]), frac(*s["mixed_retr"]), frac(*s["sql_exec"]), frac(s["oos_ok"], s["n_oos"]), s["false_ref"]]


e2e_table = t([cell(k) for k in RUNS],
              ["การรัน", "การตั้งค่า", "ตอบถูก<br>(จาก 29)", "ยืนยันด้วยแหล่งที่มา", "หลักฐานเอกสาร<br>ที่ดึงได้ (12)", "หลักฐานผสม<br>ที่ดึงได้ (6)",
               "SQL ตรงเฉลย<br>(11)", "ปฏิเสธข้อ<br>นอกขอบเขต (7)", "ปฏิเสธ<br>ผิดข้อ"], "num small")

type_rows = []
for k in ("BARE", "RAG", "A", "B2"):
    ty = S[k]["type"]
    type_rows.append([RUNS[k][0], *[frac(*ty[x]) for x in ("doc", "sql", "mixed", "oos")]])
type_table = t(type_rows, ["การรัน", "doc (12)", "sql (11)", "mixed (6)", "oos (7)"], "num2")


def bar_chart() -> str:
    order = ["BARE", "RAG", "A", "B", "B2", "C", "C2"]
    colour = {"BARE": "#B4B2A9", "RAG": "#B4B2A9", "A": "#F0997B", "B": "#1D9E75", "B2": "#1D9E75", "C": "#7F77DD", "C2": "#7F77DD"}
    w, left, bar_h, gap, top = 560, 190, 11, 10, 28
    h = top + len(order) * (2 * bar_h + gap) + 28
    scale = (w - left - 60) / 100
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" role="img" font-family="Leelawadee UI, Tahoma, sans-serif">',
           '<title>ร้อยละของคำถามที่ตอบได้ ที่ตอบถูกในแต่ละการรัน</title>']
    for pctv in (0, 25, 50, 75, 100):
        x = left + pctv * scale
        out.append(f'<line x1="{x}" y1="{top - 8}" x2="{x}" y2="{h - 26}" stroke="#E5E3DB" stroke-width="1"/>')
        out.append(f'<text x="{x}" y="{top - 14}" font-size="10" text-anchor="middle" fill="#5F5E5A">{pctv}%</text>')
    for i, k in enumerate(order):
        y = top + i * (2 * bar_h + gap)
        s = S[k]
        cp, vp = 100 * s["correct"] / s["n_in"], 100 * s["verified"] / s["n_in"]
        out.append(f'<text x="{left - 8}" y="{y + bar_h}" font-size="11" text-anchor="end" fill="#2C2C2A">{html.escape(RUNS[k][0].split("  ")[0] if k in ("A","B","B2","C","C2") else RUNS[k][0].replace("Baseline: ", ""))}</text>')
        out.append(f'<rect x="{left}" y="{y}" width="{cp * scale:.1f}" height="{bar_h}" fill="{colour[k]}"/>')
        out.append(f'<text x="{left + cp * scale + 5:.1f}" y="{y + bar_h - 2}" font-size="10" fill="#2C2C2A">ตอบถูก {s["correct"]}/{s["n_in"]} ({cp:.0f}%)</text>')
        out.append(f'<rect x="{left}" y="{y + bar_h + 2}" width="{vp * scale:.1f}" height="{bar_h - 2}" fill="{colour[k]}" opacity="0.45"/>')
        out.append(f'<text x="{left + vp * scale + 5:.1f}" y="{y + 2 * bar_h - 1}" font-size="9.5" fill="#5F5E5A">ยืนยันด้วยแหล่งที่มา {s["verified"]}/{s["n_in"]}</text>')
    out.append("</svg>")
    return "".join(out)


arch = (ROOT / "docs" / "architecture.svg").read_text(encoding="utf-8")
arch = arch[arch.index("<svg"):].replace('width="680" height="620"', 'width="100%"', 1)

today = dt.date.today()
tokens = {
    "NAME": NAME, "STUDENT_ID": STUDENT_ID, "GITHUB_URL": GITHUB_URL, "LIVE_URL": LIVE_URL,
    "DATE": f"{today.day} {THAI_MONTHS[today.month - 1]} {today.year + 543}",
    "ARCH_SVG": arch, "DATASET_TABLE": dataset_table, "REVIEW_NOTE": review_note,
    "RETRIEVAL_TABLE": retrieval_table, "FLOOR_TABLE": floor_table,
    "FLOOR_STATS": f"คะแนนความคล้าย (cosine) อันดับ 1 ต่ำสุดของข้อที่ตอบได้คือ {min(in_scope_top1):.2f} ส่วนสูงสุดของข้อนอกขอบเขตคือ {max(oos_top1):.2f}",
    "E2E_TABLE": e2e_table, "E2E_CHART": bar_chart(), "TYPE_TABLE": type_table,
    "N_TOTAL": str(n_total),
}
page = (HERE / "report.template.html").read_text(encoding="utf-8")
for key, value in tokens.items():
    page = page.replace(f"[[{key}]]", value)
unfilled = [k for k in ("[[" + x + "]]" for x in tokens) if k in page]
assert not unfilled, f"unfilled tokens: {unfilled}"
import re
stray = re.findall(r"\[\[[A-Z_0-9]+\]\]", page)
assert not stray, f"template tokens with no value: {stray}"
OUT_HTML.write_text(page, encoding="utf-8")


def browser() -> str:
    for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe"):
        if Path(p).exists():
            return p
    found = shutil.which("chromium") or shutil.which("google-chrome") or shutil.which("msedge")
    if not found:
        sys.exit("no Edge/Chrome found: open report.html in a browser and print it to PDF instead")
    return found


with tempfile.TemporaryDirectory() as profile:
    cmd = [browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={profile}",
           f"--print-to-pdf={OUT_PDF}", OUT_HTML.as_uri()]
    subprocess.run(cmd, check=True, timeout=180, capture_output=True)

from pypdf import PdfReader  # noqa: E402

pages = len(PdfReader(str(OUT_PDF)).pages)
print(f"wrote {OUT_PDF.name}: {pages} pages" + ("" if 4 <= pages <= 8 else "   <-- the assignment asks for 4-8 pages"))
