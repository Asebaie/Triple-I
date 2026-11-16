from __future__ import annotations

import math
import random
import threading
import time

import httpx

import runtime
from config import (
    RERANK_RECHECK,
    API_BASE,
    EMBED_KEY,
    EMB_MODEL,
    LLM_KEY,
    LLM_MODEL,
    MAX_TOKENS,
    MAX_WORKERS,
    RERANKER_MODEL,
    RERANK_KEY,
    TEMPERATURE,
    log,
    remaining,
    request_remaining,
)
from runtime import llm_seems_down
from textproc import THINK_RE


class APIError(Exception):
    def __init__(self, msg: str, transient: bool = False, status: int | None = None, detail: str = "") -> None:
        super().__init__(msg)
        self.transient = transient
        self.status = status
        self.detail = detail


RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}


_COOL_LOCK = threading.Lock()
_COOL_UNTIL = [0.0]


def set_cooldown(sec: float) -> None:
    with _COOL_LOCK:
        _COOL_UNTIL[0] = max(_COOL_UNTIL[0], time.monotonic() + sec)


def wait_cooldown() -> None:
    while True:
        with _COOL_LOCK:
            d = _COOL_UNTIL[0] - time.monotonic()
        if d <= 0 or remaining() < 8:
            return
        time.sleep(min(d, 5.0))


class Gate:

    def __init__(self, limit: int) -> None:
        self.max = max(1, limit)
        self.limit = self.max
        self.active = 0
        self.ok = 0
        self.cv = threading.Condition()

    def acquire(self) -> None:
        with self.cv:
            while self.active >= self.limit and remaining() > 8:
                self.cv.wait(timeout=0.5)
            self.active += 1

    def release(self, status: int) -> None:
        with self.cv:
            self.active -= 1
            if status == 429:
                if self.limit > 1:
                    self.limit -= 1
                    log(f"Прокси ограничивает частоту — одновременных запросов теперь {self.limit}")
                self.ok = 0
            elif 200 <= status < 300:
                self.ok += 1
                if self.ok >= 10 and self.limit < self.max:
                    self.limit += 1
                    self.ok = 0
            self.cv.notify_all()


GATE = Gate(MAX_WORKERS)


def post_json(
    path: str, key: str, payload: dict, timeout: float = 90.0, attempts: int = 3, max_wait: float = 60.0
) -> dict:
    if not API_BASE or not key:
        raise APIError("нет LLM_BASE_URL или ключа")
    start = time.monotonic()
    fails = 0
    last = "?"
    while True:
        wait_cooldown()
        t = min(timeout, remaining() - 3, request_remaining() - 1)
        if t < 3:
            raise APIError(f"не осталось времени ({last})", transient=True)
        GATE.acquire()
        status = -1
        t_req = time.monotonic()
        try:
            with httpx.Client(timeout=httpx.Timeout(t, connect=min(15.0, t))) as http:
                r = http.post(API_BASE + path, headers={"Authorization": "Bearer " + key}, json=payload)
            status = r.status_code
        except httpx.TransportError as exc:
            last = type(exc).__name__
            r = None
        finally:
            GATE.release(status)
        kind = path.strip("/").split("/")[-1]
        sec = time.monotonic() - t_req
        if r is None:
            runtime.CALLS.attempt(kind, "timeout" if "Timeout" in last else "transport", sec)
        elif status >= 300:
            runtime.CALLS.attempt(
                kind, "http_429" if status == 429 else ("http_5xx" if status >= 500 else f"http_{status}"), sec
            )
        if r is not None:
            if status < 300:
                try:
                    data = r.json()
                except ValueError:
                    last = "bad json"
                    runtime.CALLS.attempt(kind, "bad_json", sec)
                else:
                    runtime.CALLS.attempt(kind, "ok", sec)
                    if isinstance(data, dict):
                        runtime.CALLS.add_usage(data.get("usage"))
                    return data
            elif status == 429:
                last = "HTTP 429"
                ra = r.headers.get("retry-after", "")
                delay = float(ra) if ra.replace(".", "", 1).isdigit() else 1.0 + random.random()
                set_cooldown(min(delay, 10.0))
                if time.monotonic() - start > max_wait:
                    raise APIError(f"HTTP 429 {path}", transient=True)
                continue
            elif status in RETRY_STATUS:
                last = f"HTTP {status}"
            else:
                raise APIError(f"HTTP {status} {path}", transient=False, status=status, detail=r.text[:300])
        fails += 1
        if fails >= attempts or time.monotonic() - start > max_wait + timeout:
            raise APIError(f"{last} {path}", transient=True)
        time.sleep(
            min(2 ** (fails - 1) + random.random(), 10.0, max(0.0, remaining() - 5), max(0.0, request_remaining() - 1))
        )


def chat(messages: list[dict], max_tokens: int = MAX_TOKENS, timeout: float = 90.0) -> str:
    while llm_seems_down():
        if remaining() < 30 or request_remaining() < 5:
            raise APIError("модель недоступна до конца отведённого времени", transient=True)
        time.sleep(2.0)
    try:
        out = _chat(messages, max_tokens, timeout)
    except APIError as exc:
        if exc.transient:
            with runtime.FLAGS.lock:
                runtime.FLAGS.chat_fail_streak += 1
        raise
    with runtime.FLAGS.lock:
        runtime.FLAGS.chat_fail_streak = 0
    return out


def _chat(messages: list[dict], max_tokens: int, timeout: float) -> str:
    tokens = max_tokens
    for step in range(2):
        try:
            data = post_json(
                "/chat/completions",
                LLM_KEY,
                {
                    "model": LLM_MODEL,
                    "messages": messages,
                    "temperature": TEMPERATURE,
                    "max_tokens": tokens,
                },
                timeout=timeout,
                attempts=4,
            )
        except APIError:
            if step == 0:
                raise
            break
        try:
            choice = data["choices"][0]
            content = choice.get("message", {}).get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError):
            raise APIError("неожиданный формат ответа chat", transient=True)
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        content = THINK_RE.sub("", content).strip()
        if content:
            if choice.get("finish_reason") == "length":
                cut = max(content.rfind(". "), content.rfind(".\n"))
                if cut > len(content) // 3:
                    content = content[: cut + 1]
            return content
        if choice.get("finish_reason") == "length" and tokens < 4096 and step == 0:
            tokens = min(4096, tokens * 2)
            continue
        break
    raise APIError("пустой ответ модели", transient=True)


def embed(texts: list[str]) -> list[list[float]]:
    data = post_json(
        "/embeddings",
        EMBED_KEY,
        {"model": EMB_MODEL, "input": texts, "encoding_format": "float"},
        timeout=20,
        attempts=2,
        max_wait=30,
    )
    items = data.get("data")
    if not isinstance(items, list) or len(items) != len(texts):
        raise APIError("неожиданный формат embeddings")
    out: list[list[float] | None] = [None] * len(texts)
    try:
        for pos, it in enumerate(items):
            i = int(it.get("index", pos))
            v = [float(x) for x in it["embedding"]]
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out[i] = [x / n for x in v]
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise APIError(f"неожиданный формат embeddings: {type(exc).__name__}") from exc
    if any(v is None for v in out):
        raise APIError("получены не все embeddings")
    return out


def _rerank_disable(permanent: bool, why: str) -> None:
    f = runtime.FLAGS
    f.rerank_off = True
    if permanent:
        f.rerank_off_permanent = True
        f.rerank_off_at = None
    else:
        f.rerank_off_at = time.monotonic()
        with runtime.PROG.lock:
            f.rerank_off_q = runtime.PROG.finished
    f.stats["rerank_disabled"] = f.stats.get("rerank_disabled", 0) + 1
    log(f"Реранкер отключён {'навсегда' if permanent else 'временно'}: {why}")


def _rerank_recheck() -> None:
    f = runtime.FLAGS
    with f.lock:
        if not f.rerank_off or f.rerank_off_permanent or f.rerank_off_at is None:
            return
        with runtime.PROG.lock:
            done = runtime.PROG.finished
        if time.monotonic() - f.rerank_off_at < 120 and done - f.rerank_off_q < 10:
            return
        f.rerank_off = False
        f.rerank_off_at = None
        f.rerank_streak = 0
        f.stats["rerank_recheck"] = f.stats.get("rerank_recheck", 0) + 1
    runtime.TIMING.drop("rerank")
    log("Реранкер: повторная проба")


def rerank(query: str, docs: list[str], top_n: int) -> list[tuple[int, float]] | None:
    if RERANK_RECHECK:
        _rerank_recheck()
    if not RERANKER_MODEL or runtime.FLAGS.rerank_off or not docs:
        return None
    t0 = time.monotonic()
    try:
        data = post_json(
            "/rerank",
            RERANK_KEY,
            {"model": RERANKER_MODEL, "query": query, "documents": docs, "top_n": min(top_n, len(docs))},
            timeout=12,
            attempts=2,
            max_wait=15,
        )
        items = data.get("results") or data.get("data") or []
        res = []
        for pos, it in enumerate(items):
            i = int(it.get("index", pos))
            sc = it.get("relevance_score", it.get("score", 0.0))
            if 0 <= i < len(docs):
                res.append((i, float(sc or 0.0)))
        if not res:
            raise APIError("пустой ответ rerank")
        runtime.TIMING.add("rerank", time.monotonic() - t0)
        with runtime.FLAGS.lock:
            runtime.FLAGS.rerank_streak = 0
        if not runtime.FLAGS.rerank_logged:
            runtime.FLAGS.rerank_logged = True
            log(f"Реранкер работает, поля ответа: {sorted(data.keys())}, {time.monotonic() - t0:.1f} c")
        if runtime.TIMING.get("rerank", 0.0) > 6.0:
            if RERANK_RECHECK:
                with runtime.FLAGS.lock:
                    _rerank_disable(False, "слишком медленный")
            else:
                runtime.FLAGS.rerank_off = True
                log("Реранкер отключён: слишком медленный")
        runtime.FLAGS.inc("rerank_ok")
        return res
    except (APIError, ValueError, TypeError, KeyError, AttributeError) as exc:
        with runtime.FLAGS.lock:
            runtime.FLAGS.rerank_streak += 1
            if not (isinstance(exc, APIError) and exc.transient):
                runtime.FLAGS.rerank_fail += 1
            if RERANK_RECHECK:
                if runtime.FLAGS.rerank_fail >= 2:
                    _rerank_disable(True, "неустранимые ошибки")
                elif runtime.FLAGS.rerank_streak >= 2:
                    _rerank_disable(False, "сбои подряд")
            else:
                slow = "Timeout" in str(exc)
                if slow or runtime.FLAGS.rerank_streak >= 3 or runtime.FLAGS.rerank_fail >= 2:
                    runtime.FLAGS.rerank_off = True
        log(f"Реранкер: ошибка ({exc}){'; отключён' if runtime.FLAGS.rerank_off else ''}")
        return None
