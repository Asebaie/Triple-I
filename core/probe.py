from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time

import runtime
from config import (
    API_BASE,
    DATA_DIR,
    EMB_MODEL,
    LLM_KEY,
    LLM_MODEL,
    MAX_TOKENS,
    MAX_WORKERS,
    PROBE,
    PROBE_RERANK,
    RERANK_KEY,
    RERANKER_MODEL,
    T0,
    TEMPERATURE,
    TIME_BUDGET,
    env,
    log,
    remaining,
)
from kb import find_kb, kb_hash
from llm import APIError, _chat, post_json
from dialog import parse_dialog


def log_env() -> None:
    present = {
        k: bool(os.getenv(k))
        for k in (
            "LLM_BASE_URL",
            "OPENAI_BASE_URL",
            "LLM_API_KEY",
            "OPENAI_API_KEY",
            "LLM_MODEL",
            "OPENAI_MODEL",
            "GEN_MODEL",
        )
    }
    log(f"Переменные: {present}")
    log(
        f"Модель={LLM_MODEL} EMB={EMB_MODEL} RERANK={RERANKER_MODEL} MAX_WORKERS={MAX_WORKERS} "
        f"TEMPERATURE={TEMPERATURE} MAX_TOKENS(env)={env('MAX_TOKENS')} TOP_K={env('TOP_K')} "
        f"RERANKER_TOP_N={env('RERANKER_TOP_N')} CHUNKER={env('CHUNKER')} DATA_DIR={DATA_DIR}"
    )


def wait_for_llm() -> bool:
    delay, attempt = 2.0, 0
    probe = [{"role": "user", "content": "Ответь одним словом: готов."}]
    while remaining() > 150:
        attempt += 1
        t0 = time.monotonic()
        try:
            _chat(probe, MAX_TOKENS, 90.0)
            log(f"Модель отвечает (проба {attempt}, {time.monotonic() - t0:.1f} c)")
            return True
        except APIError as exc:
            if not exc.transient:
                log(f"Проба модели: отказ без повтора ({exc}) — продолжаем с уменьшением запроса")
                return False
            log(f"Проба модели {attempt}: нет ответа ({exc}), ждём {delay:.0f} c")
        time.sleep(min(delay, max(0.0, remaining() - 150)))
        delay = min(delay * 2, 20.0)
    log("Модель так и не ответила на пробу — продолжаем, вопросы будут ждать её сами")
    return False


ENV_PRESENT = [
    "LLM_BASE_URL",
    "OPENAI_BASE_URL",
    "BASE_URL",
    "LLM_API_KEY",
    "OPENAI_API_KEY",
    "EMBEDDER_API_KEY",
    "RERANKER_API_KEY",
    "LLM_MODEL",
    "OPENAI_MODEL",
    "GEN_MODEL",
    "DATA_DIR",
    "PYTHONPATH",
]
ENV_VALUES = [
    "LLM_MODEL",
    "OPENAI_MODEL",
    "GEN_MODEL",
    "EMB_MODEL",
    "RERANKER_MODEL",
    "TEMPERATURE",
    "MAX_TOKENS",
    "MAX_WORKERS",
    "TOP_K",
    "RERANKER_TOP_N",
    "CHUNKER",
]


def _secrets() -> list[str]:
    vals = [API_BASE, LLM_KEY, RERANK_KEY] + [
        os.getenv(k, "")
        for k in (
            "LLM_BASE_URL",
            "OPENAI_BASE_URL",
            "BASE_URL",
            "LLM_API_KEY",
            "OPENAI_API_KEY",
            "EMBEDDER_API_KEY",
            "RERANKER_API_KEY",
        )
    ]
    return sorted({v for v in vals if v and len(v) >= 6}, key=len, reverse=True)


def emit(title: str, data: dict) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=1, default=str)
    for sec in _secrets():
        text = text.replace(sec, "<скрыто>")
    print(f"=== {title} ===\n{text}\n=== /{title} ===", file=sys.stderr, flush=True)


def _memory() -> dict:
    out = {}
    try:
        out["phys_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        pass
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = open(path).read().strip()
            out["cgroup_limit_gb"] = raw if not raw.isdigit() else round(int(raw) / 2**30, 1)
            break
        except OSError:
            continue
    return out


def _rerank_format() -> dict:
    if not PROBE_RERANK:
        return {"checked": False, "reason": "SOLUTION_PROBE_RERANK не включён"}
    if not RERANKER_MODEL or not API_BASE:
        return {"checked": False}
    docs = ["Договор поставки мебели.", "Срок испытания работника.", "Ставка налога на прибыль."]
    t0 = time.monotonic()
    try:
        data = post_json(
            "/rerank",
            RERANK_KEY,
            {"model": RERANKER_MODEL, "query": "налог на прибыль", "documents": docs, "top_n": 3},
            timeout=12,
            attempts=1,
            max_wait=12,
        )
    except APIError as exc:
        return {"checked": True, "ok": False, "error": str(exc)[:120], "sec": round(time.monotonic() - t0, 2)}
    items = data.get("results") or data.get("data") or []
    scores = [it.get("relevance_score", it.get("score")) for it in items if isinstance(it, dict)]
    num = [float(x) for x in scores if isinstance(x, (int, float))]
    return {
        "checked": True,
        "ok": True,
        "sec": round(time.monotonic() - t0, 2),
        "keys": sorted(data.keys()),
        "item_keys": sorted(items[0].keys()) if items and isinstance(items[0], dict) else [],
        "n_items": len(items),
        "score_field": next((k for k in ("relevance_score", "score") if items and k in items[0]), None),
        "score_types": sorted({type(x).__name__ for x in scores}),
        "score_min": min(num) if num else None,
        "score_max": max(num) if num else None,
        "order_by_index": [it.get("index") for it in items if isinstance(it, dict)],
    }


def platform_probe(kb, questions: list, index, llm_ready, dense_sec: float | None) -> None:
    if not PROBE:
        return
    try:
        kb_path = find_kb()
        dialogs = sum(bool(parse_dialog(q)[1]) for _, q in questions)
        lens = [len(q) for _, q in questions] or [0]
        dim = len(index.vecs[0]) if getattr(index, "vecs", None) else None
        data = {
            "elapsed_sec": round(time.monotonic() - T0, 1),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu": os.cpu_count(),
            "memory": _memory(),
            "time_budget_sec": TIME_BUDGET,
            "env_present": {k: bool(os.getenv(k)) for k in ENV_PRESENT},
            "env_values": {k: os.getenv(k) for k in ENV_VALUES if os.getenv(k) is not None},
            "effective": {
                "LLM_MODEL": LLM_MODEL,
                "EMB_MODEL": EMB_MODEL,
                "RERANKER_MODEL": RERANKER_MODEL,
                "MAX_WORKERS": MAX_WORKERS,
                "TEMPERATURE": TEMPERATURE,
                "MAX_TOKENS": MAX_TOKENS,
                "DATA_DIR_is_default": str(DATA_DIR) == "data",
            },
            "kb": {
                "found": kb_path is not None,
                "format": None if kb_path is None else ("dir" if kb_path.is_dir() else "zip"),
                "file": kb_path.name if kb_path else None,
                "bytes": kb_path.stat().st_size if kb_path and kb_path.is_file() else None,
                "sha256": kb_hash(kb),
                "docs": len(kb.docs),
                "company": len(kb.company),
                "laws": len(kb.laws),
                "chunks": len(kb.chunks),
                "removed_injections": sum(len(d.removed) for d in kb.docs),
                "prebuilt_index": getattr(index, "prebuilt", None),
            },
            "questions": {
                "n": len(questions),
                "dialogs": dialogs,
                "dialog_share": round(dialogs / len(questions), 3) if questions else 0,
                "chars_mean": round(statistics.mean(lens)),
                "chars_max": max(lens),
            },
            "llm": {"ready": llm_ready, "usage_keys": sorted(runtime.CALLS.usage_keys)},
            "embeddings": {"dense": dim is not None, "dim": dim, "build_sec": dense_sec},
            "rerank": _rerank_format(),
        }
        emit("PLATFORM PROBE", data)
    except Exception as exc:
        log(f"PLATFORM PROBE не собран: {type(exc).__name__}")


def _pct(v: list[float], p: float) -> float | None:
    if not v:
        return None
    v = sorted(v)
    k = (len(v) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return round(v[lo] + (v[hi] - v[lo]) * (k - lo), 2)


def run_stats(n_questions: int, failed: int, rounds: int, levels: list | None = None) -> None:
    if not PROBE:
        return
    try:
        c = runtime.CALLS
        with c.lock:
            calls = {
                kind: {
                    "attempts": k["attempts"],
                    **k["outcomes"],
                    "p50_sec": _pct(k["lat"], 0.5),
                    "p95_sec": _pct(k["lat"], 0.95),
                    "max_sec": round(max(k["lat"]), 2) if k["lat"] else None,
                }
                for kind, k in c.kinds.items()
            }
            usage = dict(c.usage)
        usage["cache_share"] = (
            round(usage["cached_tokens"] / usage["prompt_tokens"], 3) if usage["prompt_tokens"] else None
        )
        f = runtime.FLAGS
        emit(
            "RUN STATS",
            {
                "elapsed_sec": round(time.monotonic() - T0, 1),
                "questions": n_questions,
                "fallback_answers": failed,
                "retry_rounds": rounds,
                "calls": calls,
                "usage": usage,
                "stats": dict(f.stats),
                "answer_levels": (
                    None
                    if levels is None
                    else {
                        "4": levels.count(4),
                        "3": levels.count(3),
                        "2": levels.count(2),
                        "1": levels.count(1),
                        "0_tiny": levels.count(0),
                        "fallback": levels.count(None),
                    }
                ),
                "triggers": {
                    k: f.stats.get(k, 0)
                    for k in (
                        "soft4xx",
                        "rerank_disabled",
                        "rerank_recheck",
                        "final_pass_upgraded",
                        "repaired",
                        "trimmed",
                        "ref_trimmed",
                        "skipped",
                        "rewrite",
                        "rewrite_skipped",
                        "rewrite_failed",
                        "template_trimmed",
                    )
                },
                "flags": {
                    "dense": f.dense,
                    "rerank_off": f.rerank_off,
                    "compact": f.compact,
                    "minimal": f.minimal,
                    "tiny": f.tiny,
                },
                "timing_ema_sec": {k: round(v, 2) for k, v in runtime.TIMING.ema.items()},
            },
        )
    except Exception as exc:
        log(f"RUN STATS не собран: {type(exc).__name__}")
