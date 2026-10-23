"""The CISA KEV catalog as a SQLite table the agent can query.

Built in memory from the committed CSV on every start (1.7k rows: milliseconds), so there is
no stale database file to commit or get out of sync with the snapshot.
"""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from app.config import KEV_CSV

# CSV column -> table column. Short, readable names help a small model write correct SQL.
COLUMNS = {
    "cveID": "cve_id",
    "vendorProject": "vendor",
    "product": "product",
    "vulnerabilityName": "name",
    "dateAdded": "date_added",
    "shortDescription": "description",
    "requiredAction": "required_action",
    "dueDate": "due_date",
    "knownRansomwareCampaignUse": "ransomware_use",
    "forensicTriage": "forensic_triage",
    "notes": "notes",
    "cwes": "cwes",
}

# Shown to the model (tool description + system prompt). Keep it exact: the literal DDL
# beats a prose description because the model stops guessing column names (Lecture 11).
SCHEMA_DOC = """\
Table kev - one row per vulnerability in CISA's Known Exploited Vulnerabilities catalog:
  cve_id          TEXT  primary key, e.g. 'CVE-2024-3400'
  vendor          TEXT  e.g. 'Microsoft', 'Cisco', 'Apache'   (proper-case; use LIKE for partial matches)
  product         TEXT  e.g. 'Windows', 'Exchange Server'
  name            TEXT  vulnerability name
  date_added      TEXT  'YYYY-MM-DD' - the day CISA added it to the catalog
  description     TEXT  short description
  required_action TEXT  CISA's required remediation text
  due_date        TEXT  'YYYY-MM-DD' - federal remediation deadline
  ransomware_use  TEXT  exactly 'Known' or 'Unknown' (known use in ransomware campaigns)
  forensic_triage TEXT  exactly 'Yes' or 'No'
  notes           TEXT  links and remarks
  cwes            TEXT  comma-separated CWE ids such as 'CWE-78, CWE-79'; empty string if none
Dates are ISO text: use date_added >= '2025-01-01' or strftime('%Y', date_added) = '2025'."""


def connect_kev(csv_path: Path = KEV_CSV) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:", check_same_thread=False)
    cols = ", ".join(f"{c} TEXT" for c in COLUMNS.values())
    con.execute(f"CREATE TABLE kev({cols}, PRIMARY KEY(cve_id))")
    with open(csv_path, encoding="utf-8", newline="") as f:
        rows = [tuple((r.get(src) or "").strip() for src in COLUMNS) for r in csv.DictReader(f)]
    con.executemany(f"INSERT INTO kev VALUES ({', '.join('?' * len(COLUMNS))})", rows)
    con.commit()
    return con
