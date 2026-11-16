from __future__ import annotations

import time
from typing import TYPE_CHECKING

from answering import answer_one, answer_tiny
from config import DEBUG
from dialog import Prepared, event_text, parse_dialog, search_query, user_text
from kb import doc_ids
from llm import APIError, embed
from result import STATUSES, Result, build_context, build_warnings, classify_status, extract_citations
from retrieval import company_context, fallback_answer, maybe_rewrite, select_law, weak_search

if TYPE_CHECKING:
    from index import Index
    from kb import KB

__all__ = [
    "answer",
    "answer_one",
    "answer_tiny",
    "Prepared",
    "parse_dialog",
    "event_text",
    "user_text",
    "search_query",
    "fallback_answer",
    "select_law",
    "company_context",
    "maybe_rewrite",
    "weak_search",
    "Result",
    "STATUSES",
    "extract_citations",
    "classify_status",
    "build_warnings",
    "build_context",
]


def answer(
    kb: KB, index: Index, qtext: str, level: int = 4, trace=None, debug: bool = False, ids: dict[int, str] | None = None
) -> Result:
    ids = ids or doc_ids(kb)
    marks: dict[str, float] = {}
    t0 = time.monotonic()

    def stage(name: str) -> None:
        marks[name] = time.monotonic()
        if trace:
            try:
                trace(name)
            except Exception:
                pass

    prep = Prepared(*parse_dialog(qtext))
    info: dict = {}
    try:
        if index.vecs is not None:
            q = search_query(prep)[:2000]
            try:
                prep.qvec = embed([q])[0]
            except APIError:
                prep.qvec = None
        text = answer_one(kb, index, qtext, prep, level, stage, info)
        status = None
        warns: list[dict] = []
    except Exception as exc:
        text = fallback_answer(kb, index, qtext)
        status = "error"
        code = "llm_unavailable" if isinstance(exc, APIError) else "internal_error"
        warns = [
            {"code": code, "message": "Модель не ответила; показан ближайший фрагмент базы знаний"},
            {"code": "fallback", "message": type(exc).__name__},
        ]
    stage("done")
    citations = extract_citations(kb, text, qtext, ids) if status != "error" else []
    context: dict = {}
    if status is None:
        status = classify_status(text, citations)
        warns = build_warnings(kb, qtext, info, citations, ids)
        try:
            context = build_context(kb, index, info, prep, qtext, ids, citations)
        except Exception:
            context = {}
    order = ["search", "generate", "verify", "done"]
    seen = [n for n in order if n in marks]
    timing = {}
    for a, b in zip(seen, seen[1:]):
        timing[a] = round((marks[b] - marks[a]) * 1000)
    timing["total"] = round((marks["done"] - t0) * 1000)
    dbg = {}
    if debug or DEBUG:
        dbg = {k: info.get(k) for k in ("level", "laws", "company", "date_note", "draft", "unverified", "prompt_chars")}
    return Result(
        text=text, status=status, citations=citations, warnings=warns, timing_ms=timing, debug=dbg, context=context
    )
