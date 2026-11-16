from __future__ import annotations

import threading
import time

from config import LEVEL_FORECAST, MAX_WORKERS, remaining


class Flags:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.chat_fail_streak = 0
        self.last_probe = 0.0
        self.rerank_fail = 0
        self.rerank_streak = 0
        self.rerank_off = False
        self.rerank_logged = False
        self.dense = False
        self.compact = False
        self.minimal = False
        self.tiny = False
        self.soft4xx_streak = 0
        self.degraded_at = -1
        self.rerank_off_at = None
        self.rerank_off_q = 0
        self.rerank_off_permanent = False
        self.stats = {
            "llm_ok": 0,
            "fallback": 0,
            "repaired": 0,
            "trimmed": 0,
            "rerank_ok": 0,
            "level4": 0,
            "level3": 0,
            "level2": 0,
            "level1": 0,
            "skipped": 0,
        }

    def inc(self, k: str) -> None:
        with self.lock:
            self.stats[k] = self.stats.get(k, 0) + 1


FLAGS = Flags()


class Timing:

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.ema: dict[str, float] = {}

    def add(self, key: str, value: float) -> None:
        with self.lock:
            old = self.ema.get(key)
            self.ema[key] = value if old is None else 0.7 * old + 0.3 * value

    def get(self, key: str, default: float) -> float:
        with self.lock:
            return self.ema.get(key, default)

    def drop(self, key: str) -> None:
        with self.lock:
            self.ema.pop(key, None)


TIMING = Timing()


def llm_seems_down() -> bool:
    with FLAGS.lock:
        if FLAGS.chat_fail_streak < 6:
            return False
        now = time.monotonic()
        if now - FLAGS.last_probe > 20:
            FLAGS.last_probe = now
            return False
        return True


class Progress:
    def __init__(self) -> None:
        self.n = 1
        self.finished = 0
        self.lock = threading.Lock()


PROG = Progress()


def per_question_budget() -> float:
    with PROG.lock:
        left = max(1, PROG.n - PROG.finished)
    return max(0.0, remaining() - 15) * MAX_WORKERS / left


def question_cost(level: int) -> float:
    full = TIMING.get("chat_full", 8.0)
    comp = TIMING.get("chat_compact", full * 0.6)
    mini = TIMING.get("chat_min", comp * 0.7)
    rr = 0.0 if FLAGS.rerank_off else TIMING.get("rerank", 2.0)
    default = {4: (full + rr) * 1.3, 3: full * 1.1, 2: comp * 1.05, 1: mini, 0: mini * 0.5}[level]
    return TIMING.get(f"q{level}", default)


def _choose_level_forecast() -> int:
    full = TIMING.get("chat_full", 8.0)
    comp = TIMING.get("chat_compact", full * 0.6)
    mini = TIMING.get("chat_min", comp * 0.7)
    rem = remaining() - 15
    if rem < min(mini, 15.0):
        return 0
    if FLAGS.minimal:
        return 1
    with PROG.lock:
        left = max(1, PROG.n - PROG.finished)
    allowed = (2, 1) if FLAGS.compact else (4, 3, 2, 1)
    for lv in allowed:
        if left * question_cost(lv) / MAX_WORKERS * 1.2 <= rem:
            return lv
    return allowed[-1]


def choose_level() -> int:
    if LEVEL_FORECAST:
        return _choose_level_forecast()
    full = TIMING.get("chat_full", 8.0)
    comp = TIMING.get("chat_compact", full * 0.6)
    mini = TIMING.get("chat_min", comp * 0.7)
    if remaining() - 15 < min(mini, 15.0):
        return 0
    per_q = per_question_budget()
    if FLAGS.minimal:
        return 1
    if FLAGS.compact:
        return 2 if per_q >= comp * 1.15 else 1
    rr = 0.0 if FLAGS.rerank_off else TIMING.get("rerank", 2.0)
    if per_q >= (full + rr) * 1.6:
        return 4
    if per_q >= full * 1.2:
        return 3
    if per_q >= comp * 1.15:
        return 2
    return 1


class CallStats:

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.kinds: dict[str, dict] = {}
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "with_usage": 0}
        self.usage_keys: set[str] = set()

    def attempt(self, kind: str, outcome: str, sec: float) -> None:
        with self.lock:
            k = self.kinds.setdefault(kind, {"attempts": 0, "outcomes": {}, "lat": []})
            k["attempts"] += 1
            k["outcomes"][outcome] = k["outcomes"].get(outcome, 0) + 1
            if outcome == "ok":
                k["lat"].append(sec)

    def add_usage(self, usage) -> None:
        if not isinstance(usage, dict):
            return
        with self.lock:
            self.usage_keys.update(usage.keys())
            details = usage.get("prompt_tokens_details")
            if isinstance(details, dict):
                self.usage_keys.update(f"prompt_tokens_details.{x}" for x in details)
            self.usage["with_usage"] += 1
            self.usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            self.usage["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            hit = usage.get("prompt_cache_hit_tokens")
            if hit is None and isinstance(details, dict):
                hit = details.get("cached_tokens")
            self.usage["cached_tokens"] += int(hit or 0)


CALLS = CallStats()
