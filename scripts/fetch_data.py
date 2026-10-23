"""Fetch the project's raw data into data/ (run once; the results are committed).

  uv run python scripts/fetch_data.py            # download what is missing
  uv run python scripts/fetch_data.py --force    # re-download everything

Two sources, both redistributable:
  1. CISA KEV catalog (CSV)            -> data/raw/known_exploited_vulnerabilities.csv  (CC0 1.0)
  2. NIST security publications (PDF)  -> data/docs/<slug>.txt  (US-government works; see SOURCES.md)

The KEV file changes weekly, so the committed snapshot is what the evaluation gold is
computed against. Re-fetching it will change the numbers - that is why it is committed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import logging
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import httpx
from pypdf import PdfReader

logging.getLogger("pypdf").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DOCS = ROOT / "data" / "docs"
PDF_CACHE = RAW / "pdf"  # not committed (see .gitignore): the .txt extracts are

# An honest, identifying User-Agent. (cisa.gov HTML pages refuse non-browser clients with
# 403; we do not work around that - the KEV data file below is the published feed.)
HEADERS = {"User-Agent": "cyber-advisor-capstone/0.1 (student project; polite, low-rate)"}

KEV_URL = "https://www.cisa.gov/sites/default/files/csv/known_exploited_vulnerabilities.csv"
KEV_FILE = RAW / "known_exploited_vulnerabilities.csv"

NIST = "https://nvlpubs.nist.gov/nistpubs/"
# slug -> (publication id, title, path under NIST). IR 8374 was dropped: it is marked Withdrawn.
PUBLICATIONS: dict[str, tuple[str, str, str]] = {
    "sp800-61r3": ("NIST SP 800-61 Rev. 3", "Incident Response Recommendations and Considerations for Cybersecurity Risk Management",
                   "SpecialPublications/NIST.SP.800-61r3.pdf"),
    "sp800-40r4": ("NIST SP 800-40 Rev. 4", "Guide to Enterprise Patch Management Planning",
                   "SpecialPublications/NIST.SP.800-40r4.pdf"),
    "csf-2.0": ("NIST CSWP 29", "The NIST Cybersecurity Framework (CSF) 2.0",
                "CSWP/NIST.CSWP.29.pdf"),
    "sp800-218": ("NIST SP 800-218", "Secure Software Development Framework (SSDF) Version 1.1",
                  "SpecialPublications/NIST.SP.800-218.pdf"),
    "sp800-83r1": ("NIST SP 800-83 Rev. 1", "Guide to Malware Incident Prevention and Handling for Desktops and Laptops",
                   "SpecialPublications/NIST.SP.800-83r1.pdf"),
    "sp800-115": ("NIST SP 800-115", "Technical Guide to Information Security Testing and Assessment",
                  "Legacy/SP/nistspecialpublication800-115.pdf"),
    "sp800-30r1": ("NIST SP 800-30 Rev. 1", "Guide for Conducting Risk Assessments",
                   "Legacy/SP/nistspecialpublication800-30r1.pdf"),
    "sp800-92": ("NIST SP 800-92", "Guide to Computer Security Log Management",
                 "Legacy/SP/nistspecialpublication800-92.pdf"),
    "sp800-207": ("NIST SP 800-207", "Zero Trust Architecture",
                  "SpecialPublications/NIST.SP.800-207.pdf"),
    "sp800-150": ("NIST SP 800-150", "Guide to Cyber Threat Information Sharing",
                  "SpecialPublications/NIST.SP.800-150.pdf"),
    "sp800-190": ("NIST SP 800-190", "Application Container Security Guide",
                  "SpecialPublications/NIST.SP.800-190.pdf"),
    "sp800-128": ("NIST SP 800-128", "Guide for Security-Focused Configuration Management of Information Systems",
                  "SpecialPublications/NIST.SP.800-128.pdf"),
}


def fetch_kev(client: httpx.Client, force: bool) -> dict:
    if force or not KEV_FILE.exists():
        r = client.get(KEV_URL)
        r.raise_for_status()
        RAW.mkdir(parents=True, exist_ok=True)
        KEV_FILE.write_bytes(r.content)
        print(f"downloaded KEV: {len(r.content) // 1024} KB")
    else:
        print("KEV already present (use --force to refresh)")
    data = KEV_FILE.read_bytes()
    n_rows = data.count(b"\nCVE-")
    snapshot = {
        "file": KEV_FILE.name, "source": KEV_URL, "license": "CC0 1.0 (https://www.cisa.gov/sites/default/files/licenses/kev/license.txt)",
        "fetched": dt.date.today().isoformat(), "sha256": hashlib.sha256(data).hexdigest(), "approx_rows": n_rows,
    }
    (RAW / "KEV_SNAPSHOT.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return snapshot


# ---- PDF -> clean text ---------------------------------------------------------------

_PAGE_NUM = re.compile(r"^\s*(?:page\s+)?(?:[ivxlc]+|\d{1,3})\s*$", re.I)
# horizontal rules ("_____") and per-page footers such as "CHAPTER 2 PAGE 11" / "APPENDIX F PAGE F-3"
_RULE_OR_FOOTER = re.compile(r"^(?:[_\-=\s]{5,}|(?:chapter|appendix)\s+\S+\s+page\s+\S+)$", re.I)
_PRIVATE_USE = re.compile(r"[-]")  # glyphs from symbol fonts: carry no meaning in text
_ASCII_PUNCT = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',   # curly quotes
    "–": "-", "—": "-", "−": "-",                  # dashes, minus
    "•": "-", "●": "-", "‣": "-",                  # bullets
    " ": " ",
})


def normalize_text(text: str) -> str:
    """NFKC (ligatures like 'fi' -> 'fi'), typographic punctuation -> ASCII, symbol-font junk out.
    Keeps retrieval and exact-phrase evidence matching independent of how the PDF was typeset."""
    return _PRIVATE_USE.sub("", unicodedata.normalize("NFKC", text)).translate(_ASCII_PUNCT)


def clean_pages(pages: list[str]) -> list[str]:
    """Drop running headers/footers (lines that repeat on many pages) and bare page
    numbers, and undo end-of-line hyphenation. Extraction from PDF is never perfect;
    this just removes the worst noise so it does not pollute retrieval."""
    pages = [normalize_text(p) for p in pages]
    # squash runs of spaces first: a running header padded with spaces to the right margin
    # is the same header on every page, but only once the padding is gone
    lines_per_page = [[" ".join(ln.split()) for ln in p.splitlines()] for p in pages]
    freq: Counter[str] = Counter()
    for lines in lines_per_page:
        freq.update({ln for ln in lines if 3 < len(ln) < 160})
    threshold = max(4, int(0.3 * len(pages)))
    boilerplate = {ln for ln, c in freq.items() if c >= threshold}
    cleaned = []
    for lines in lines_per_page:
        keep = [ln for ln in lines
                if ln and ln not in boilerplate and not _PAGE_NUM.match(ln) and not _RULE_OR_FOOTER.match(ln)]
        text = "\n".join(keep)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # re-join hyphenated words
        text = re.sub(r"(?<![.:;!?\n])\n(?!\n)", " ", text)   # soft-wrap -> space
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        cleaned.append(text)
    return cleaned


def fetch_publications(client: httpx.Client, force: bool) -> list[dict]:
    DOCS.mkdir(parents=True, exist_ok=True)
    PDF_CACHE.mkdir(parents=True, exist_ok=True)
    manifest = []
    for slug, (pub_id, title, path) in PUBLICATIONS.items():
        pdf_path = PDF_CACHE / f"{slug}.pdf"
        txt_path = DOCS / f"{slug}.txt"
        if force or not pdf_path.exists():
            r = client.get(NIST + path)
            r.raise_for_status()
            if "pdf" not in r.headers.get("content-type", ""):
                raise RuntimeError(f"{slug}: expected a PDF, got {r.headers.get('content-type')}")
            pdf_path.write_bytes(r.content)
            time.sleep(1.0)  # be polite
        if force or not txt_path.exists():
            reader = PdfReader(io.BytesIO(pdf_path.read_bytes()))
            pages = clean_pages([(p.extract_text() or "") for p in reader.pages])
            txt_path.write_text("\f".join(pages), encoding="utf-8")  # \f separates pages
        n_pages = txt_path.read_text(encoding="utf-8").count("\f") + 1
        manifest.append({"slug": slug, "id": pub_id, "title": title, "source_url": NIST + path,
                         "pages": n_pages, "chars": txt_path.stat().st_size})
        print(f"  {slug:<12} {n_pages:>3} pages  {txt_path.stat().st_size // 1024:>4} KB  {pub_id}")
    (DOCS / "_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def write_sources_md(kev: dict, manifest: list[dict]) -> None:
    lines = [
        "# Data sources and licences", "",
        "Everything under `data/` is regenerated by `scripts/fetch_data.py`; the extracts are committed so the",
        "evaluation numbers are reproducible from the repo alone.", "",
        "## Table: CISA Known Exploited Vulnerabilities (KEV) catalog", "",
        f"- Source: <{kev['source']}> (mirror: github.com/cisagov/kev-data)",
        f"- Licence: {kev['license']}",
        f"- Snapshot fetched: {kev['fetched']}  (sha256 `{kev['sha256'][:16]}...`)",
        "- The catalog changes every week. **Evaluation gold is computed against this committed snapshot.**", "",
        "## Documents: NIST publications (`data/docs/*.txt`, pages separated by form-feed)", "",
        "NIST-authored publications are works of the US Government and are not subject to copyright in the",
        "United States (17 U.S.C. 105); each PDF's own notice governs and is preserved in the extracted text.",
        "Text was extracted with pypdf and lightly cleaned (running headers/footers and page numbers removed),",
        "so wording is the original but layout is not.", "",
        "| slug | publication | pages | source |", "|---|---|---|---|",
    ]
    for m in manifest:
        lines.append(f"| `{m['slug']}` | {m['id']}: {m['title']} | {m['pages']} | <{m['source_url']}> |")
    lines += ["", "## Deliberately excluded", "",
              "- cisa.gov HTML pages (advisories, BODs, playbooks): the site returns HTTP 403 to non-browser clients, and we do not work around that.",
              "- NIST IR 8374 (Ransomware Risk Management): its cover page marks it *Withdrawn*, so it would add stale guidance.", ""]
    (ROOT / "data" / "SOURCES.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download everything")
    args = ap.parse_args()
    with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=60) as client:
        kev = fetch_kev(client, args.force)
        print("NIST publications:")
        manifest = fetch_publications(client, args.force)
    write_sources_md(kev, manifest)
    print("done -> data/SOURCES.md")


if __name__ == "__main__":
    main()
