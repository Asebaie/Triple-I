from __future__ import annotations

import os
import sys
import threading
import time
from datetime import date
from pathlib import Path

T0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[{time.monotonic() - T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


def env(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    return default if not value or value == name else value


def env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(float(env(name)))))
    except ValueError:
        return default


def env_float(name: str, default: float, low: float = 0.0, high: float = 2.0) -> float:
    try:
        return max(low, min(high, float(env(name))))
    except ValueError:
        return default


DATA_DIR = Path(env("DATA_DIR", "data"))
API_BASE = (env("LLM_BASE_URL") or env("BASE_URL") or env("OPENAI_BASE_URL")).rstrip("/")
LLM_KEY = env("LLM_API_KEY") or env("OPENAI_API_KEY")
EMBED_KEY = env("EMBEDDER_API_KEY") or LLM_KEY
RERANK_KEY = env("RERANKER_API_KEY") or LLM_KEY
LLM_MODEL = env("LLM_MODEL") or env("GEN_MODEL") or env("OPENAI_MODEL", "deepseek/deepseek-v4.1-flash")
EMB_MODEL = env("EMB_MODEL", "baai/bge-m3")
RERANKER_MODEL = env("RERANKER_MODEL", "Qwen/Qwen3-Reranker-8B")

MAX_WORKERS = env_int("MAX_WORKERS", 2, 1, 16)
TEMPERATURE = env_float("TEMPERATURE", 0.0)
MAX_TOKENS = env_int("MAX_TOKENS", 1024, 128, 8192)
TOP_K_ENV = env_int("TOP_K", 0, 0, 200)
RERANK_N_ENV = env_int("RERANKER_TOP_N", 0, 0, 100)

TIME_BUDGET = env_int("SOLUTION_TIME_BUDGET", 16 * 60 + 30, 30, 3600)
DEADLINE = T0 + TIME_BUDGET

CANDIDATES = min(40, max(30, TOP_K_ENV))
LAW_TOP = min(14, max(8, RERANK_N_ENV))
LAW_BUDGET_CHARS = 12000
SMALL_LAW_CHARS = 7000
COMPANY_FULL_LIMIT = 45000
COMPANY_TOP_DOCS = 4
CHUNK_MAX = 1600
ANSWER_MAX_CHARS = 3000

OUTPUT_FILE = "submission.csv"
DEBUG = env("SOLUTION_DEBUG", "") == "1"
DEBUG_LOCK = threading.Lock()

PROBE = env("SOLUTION_PROBE", "1") != "0"
PROBE_RERANK = env("SOLUTION_PROBE_RERANK", "") == "1"

FEWSHOT = env("SOLUTION_FEWSHOT", "fictional")
FIX_DOCNUM = env("SOLUTION_FIX_DOCNUM", "") == "1"
CHECK_REFS = env("SOLUTION_CHECK_REFS", "") == "1"
DATE_CONDITIONS = env("SOLUTION_DATE_CONDITIONS", "") == "1"
CROSS_REFS = env("SOLUTION_CROSS_REFS", "") == "1"
ANSWER_TEMPLATE = env("SOLUTION_ANSWER_TEMPLATE", "1") != "0"
CHUNK_FIXES = env("SOLUTION_CHUNK_FIXES", "") == "1"
COMPANY_TOP = env("SOLUTION_COMPANY_TOP", "1") != "0"
CHECK_FIXES = env("SOLUTION_CHECK_FIXES", "1") != "0"
SOURCE_RULE = env("SOLUTION_SOURCE_RULE", "1") != "0"
GENERAL_POINTS = env("SOLUTION_GENERAL_POINTS", "1") != "0"
REWRITE = env("SOLUTION_REWRITE", "1") != "0"
REWRITE_PROMPT = env("SOLUTION_REWRITE_PROMPT", "1") != "0"
REWRITE_COS = 0.58
REWRITE_BM25 = 10.0
CONCISE = env("SOLUTION_CONCISE", "1") != "0"
SOFT_DEGRADE = env("SOLUTION_SOFT_DEGRADE", "1") != "0"
RERANK_RECHECK = env("SOLUTION_RERANK_RECHECK", "1") != "0"
LEVEL_FORECAST = env("SOLUTION_LEVEL_FORECAST", "1") != "0"
FINAL_PASS = env("SOLUTION_FINAL_PASS", "1") != "0"
PREBUILT_INDEX = env("SOLUTION_PREBUILT_INDEX", "1") != "0"

SOLUTION_TODAY = env("SOLUTION_TODAY")


def today() -> date:
    if SOLUTION_TODAY:
        try:
            return date.fromisoformat(SOLUTION_TODAY)
        except ValueError:
            pass
    return date.today()


def debug_dump(record: dict) -> None:
    if not DEBUG:
        return
    import json

    with DEBUG_LOCK, open("debug.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


MODE = env("SOLUTION_MODE", "batch")


def remaining() -> float:
    if MODE == "web":
        return 3600.0
    return DEADLINE - time.monotonic()


_REQUEST = threading.local()


def set_request_deadline(seconds: float | None) -> None:
    _REQUEST.deadline = None if seconds is None else time.monotonic() + seconds


def request_remaining() -> float:
    deadline = getattr(_REQUEST, "deadline", None)
    return float("inf") if deadline is None else deadline - time.monotonic()
