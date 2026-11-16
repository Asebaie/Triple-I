from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Callable

CONTRACT_VERSION = "1.1"

_CORE_MODULES = (
    "config",
    "textproc",
    "dates",
    "kb",
    "runtime",
    "llm",
    "dialog",
    "index",
    "prompts",
    "checks",
    "refs",
    "retrieval",
    "answering",
    "result",
    "pipeline",
)


class LegalAssistant:

    def __init__(
        self,
        overrides: dict | None = None,
        answer_timeout: float = 60.0,
        degrade_recheck: float = 300.0,
        debug: bool = False,
    ) -> None:
        import sys

        already = [m for m in _CORE_MODULES if m in sys.modules]
        if overrides and already:
            self._late_overrides = sorted(overrides)
        else:
            self._late_overrides = []
            for k, v in (overrides or {}).items():
                os.environ[k] = str(v)
        os.environ.setdefault("SOLUTION_MODE", "web")
        import config
        import kb as kbmod
        import pipeline
        import runtime

        if config.MODE != "web":
            for name in _CORE_MODULES + ("probe",):
                mod = sys.modules.get(name)
                if mod is not None and hasattr(mod, "MODE"):
                    mod.MODE = "web"
        self._config, self._kbmod, self._pipeline, self._runtime = config, kbmod, pipeline, runtime
        self.answer_timeout = float(answer_timeout)
        self.degrade_recheck = float(degrade_recheck)
        self.debug = debug
        self._lock = threading.Lock()
        self._kb = None
        self._index = None
        self._ids: dict[int, str] = {}
        self._by_id: dict[str, object] = {}
        self._degraded_since: float | None = None
        self._llm_ok: bool | None = None
        self._warm_error: str | None = None
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="legal-answer")

    def warmup(self) -> dict:
        with self._lock:
            if self._kb is None:
                try:
                    import index as indexmod
                    import contextlib
                    import io

                    with contextlib.redirect_stderr(io.StringIO()):
                        kb = self._kbmod.build_kb()
                        idx = indexmod.Index(kb)
                        if kb.chunks and self._config.API_BASE:
                            if not (self._config.PREBUILT_INDEX and idx.load_prebuilt().get("used")):
                                idx.build_dense(cap=60.0)
                    self._kb, self._index = kb, idx
                    self._ids = self._kbmod.doc_ids(kb)
                    self._by_id = {v: kb.docs[k] for k, v in self._ids.items()}
                    self._warm_error = None
                except Exception as exc:
                    self._warm_error = f"{type(exc).__name__}: {exc}"[:300]
        return self.health()

    def _ready(self) -> bool:
        if self._kb is None:
            self.warmup()
        return self._kb is not None and bool(self._kb.docs)

    @staticmethod
    def to_platform_text(messages: list[dict]) -> str:
        turns = [m for m in messages if m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()]
        while turns and turns[-1]["role"] != "user":
            turns.pop()
        if len(turns) == 1:
            return str(turns[0]["content"]).strip()
        return "\n".join(
            ("Пользователь: " if m["role"] == "user" else "Ассистент: ") + str(m["content"]).strip() for m in turns
        )

    def answer(self, messages: list[dict], on_stage: Callable[[str], None] | None = None):
        try:
            text = self.to_platform_text(messages or [])
        except Exception:
            text = ""
        return self.answer_text(text, on_stage)

    def answer_text(self, question_text: str, on_stage: Callable[[str], None] | None = None):
        P = self._pipeline
        t0 = time.monotonic()
        if not str(question_text or "").strip():
            return self._error("empty_question", "Пустой вопрос", t0)
        try:
            if not self._ready():
                return self._error(
                    "kb_unavailable", f"База знаний не загружена: {self._warm_error or 'нет документов'}", t0
                )
            self._recheck_degradation()
            fut = self._pool.submit(self._answer_with_deadline, question_text, on_stage)
            try:
                res = fut.result(timeout=self.answer_timeout)
            except FutureTimeout:
                fut.cancel()
                self._llm_ok = False
                return self._error("timeout", f"Ответ не получен за {self.answer_timeout:.0f} с", t0)
            if res.status != "error":
                self._llm_ok = True
            else:
                self._llm_ok = False
            res.contract_version = CONTRACT_VERSION
            return res
        except Exception as exc:
            return self._error("internal_error", type(exc).__name__, t0)

    def _answer_with_deadline(self, question_text: str, on_stage):
        self._config.set_request_deadline(self.answer_timeout + 2.0)
        try:
            return self._pipeline.answer(self._kb, self._index, question_text, 4, on_stage, self.debug, self._ids)
        finally:
            self._config.set_request_deadline(None)

    def _error(self, code: str, message: str, t0: float):
        res = self._pipeline.Result(
            text="Не удалось сформировать ответ по базе знаний. Попробуйте ещё раз позже.",
            status="error",
            warnings=[{"code": code, "message": message}],
            timing_ms={"total": round((time.monotonic() - t0) * 1000)},
        )
        res.contract_version = CONTRACT_VERSION
        return res

    def _recheck_degradation(self) -> None:
        f = self._runtime.FLAGS
        degraded = f.rerank_off or f.compact or f.minimal or f.tiny or f.chat_fail_streak >= 6
        now = time.monotonic()
        if not degraded:
            self._degraded_since = None
            return
        if self._degraded_since is None:
            self._degraded_since = now
        elif now - self._degraded_since >= self.degrade_recheck:
            with f.lock:
                f.rerank_off = f.compact = f.minimal = f.tiny = False
                f.rerank_fail = f.rerank_streak = f.chat_fail_streak = 0
            self._degraded_since = None

    def list_documents(self) -> list[dict]:
        if not self._ready():
            return []
        K = self._kbmod
        out = []
        for d in self._kb.docs:
            out.append(
                {
                    "doc_id": self._ids[d.idx],
                    "title": d.title,
                    "label": d.label,
                    "kind": d.kind,
                    "type": K.doc_type(d),
                    "status": d.status or ("действующая редакция" if d.kind == "law" else ""),
                    "valid_from": d.valid_from.isoformat() if d.valid_from else None,
                    "valid_to": d.valid_to.isoformat() if d.valid_to else None,
                    "export_date": d.export_date or None,
                    "related": [self._ids[o.idx] for o in K.related_docs(self._kb, d)] if d.kind == "company" else [],
                    "has_removed_instructions": bool(d.removed),
                }
            )
        return out

    def get_document(self, doc_id: str) -> dict | None:
        if not self._ready():
            return None
        d = self._by_id.get(doc_id)
        if d is None:
            return None
        K = self._kbmod
        markup = K.doc_markup(d)
        meta = next(x for x in self.list_documents() if x["doc_id"] == doc_id)
        return {
            **meta,
            "contract_version": CONTRACT_VERSION,
            "text": d.text,
            "clauses": [{k: c[k] for k in ("clause", "start", "end", "title")} for c in K.doc_clauses(d)],
            "edition_notes": markup["edition_notes"],
            "removed_instructions": markup["removed_instructions"],
        }

    def health(self) -> dict:
        c = self._config
        f = self._runtime.FLAGS
        kb = self._kb
        return {
            "contract_version": CONTRACT_VERSION,
            "mode": c.MODE,
            "kb_loaded": kb is not None and bool(kb.docs),
            "kb_error": self._warm_error,
            "documents": len(kb.docs) if kb else 0,
            "chunks": len(kb.chunks) if kb else 0,
            "dense_index": bool(self._index is not None and self._index.vecs is not None),
            "llm_configured": bool(c.API_BASE and c.LLM_KEY),
            "llm_last_ok": self._llm_ok,
            "model": c.LLM_MODEL,
            "rerank_enabled": not f.rerank_off,
            "degraded": {"rerank_off": f.rerank_off, "compact": f.compact, "minimal": f.minimal, "tiny": f.tiny},
            "answer_timeout_sec": self.answer_timeout,
            "late_overrides_ignored": self._late_overrides,
        }
