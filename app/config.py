"""Configuration, read from the environment (a local .env is loaded for convenience).

Nothing secret lives in the repo: OPENAI_API_KEY and OPENAI_BASE_URL come from the
environment, exactly like the course sandbox. The defaults below are the configuration the
evaluation found best (README, "Results"). To reproduce the baseline it is compared with
(the reference app's defaults), set CHUNKING=naive:800:150 RETRIEVAL=dense TOP_K=5 MIN_SCORE=0.40.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA = ROOT / "data"
DOCS_DIR = DATA / "docs"
KEV_CSV = DATA / "raw" / "known_exploited_vulnerabilities.csv"
INDEX_DIR = DATA / "index"  # derived (embeddings); rebuilt by `python -m app.ingest`
DEFAULT_BASE_URL = "https://llm.nat-d.uk/v1"  # the class proxy


@dataclass(frozen=True)
class Settings:
    chat_model: str
    embed_model: str
    chunking: str    # "naive:800:150" | "section:1200"   (see app/chunking.py)
    retrieval: str   # "dense" | "bm25" | "hybrid"
    top_k: int
    min_score: float  # cosine floor below which we refuse instead of grounding on junk
    max_steps: int    # hard cap on the agent loop
    prompt: str = "v1"  # system-prompt version (see app/agent.py): v1 = original, v2 = adds routing rules


def get_settings() -> Settings:
    env = os.environ.get
    return Settings(
        chat_model=env("CHAT_MODEL", "gemma-4-E4B-it"),
        embed_model=env("EMBED_MODEL", "bge-m3"),
        chunking=env("CHUNKING", "naive:800:150"),
        retrieval=env("RETRIEVAL", "hybrid"),
        top_k=int(env("TOP_K", "10")),
        min_score=float(env("MIN_SCORE", "0.55")),
        max_steps=int(env("MAX_STEPS", "5")),
        prompt=env("PROMPT", "v1"),
    )
