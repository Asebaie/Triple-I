from __future__ import annotations

import json
import re
import time
from typing import TYPE_CHECKING

import runtime
from config import (
    CANDIDATES,
    COMPANY_FULL_LIMIT,
    COMPANY_TOP,
    COMPANY_TOP_DOCS,
    CROSS_REFS,
    GENERAL_POINTS,
    LAW_BUDGET_CHARS,
    LAW_TOP,
    MAX_WORKERS,
    MODE,
    REWRITE_BM25,
    REWRITE_COS,
    SMALL_LAW_CHARS,
    log,
    remaining,
)
from dialog import Prepared, parse_dialog, search_query
from kb import cross_refs, related_docs
from llm import APIError, chat, embed, rerank
from prompts import SYSTEM_REWRITE, rewrite_message
from textproc import STEMMER, STOPWORDS, WORD_RE, expand_query, norm

if TYPE_CHECKING:
    from index import Index
    from kb import KB, Chunk, Doc


def fallback_answer(kb: KB, index: Index, question: str) -> str:
    ranked = index.bm25(expand_query(search_query(Prepared(*parse_dialog(question)), assistant=True)))
    for ci, _ in ranked[:1]:
        c = kb.chunks[ci]
        text = re.sub(r"\[Пометка редакции:[^\]]*\]", " ", c.text)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 450:
            text = text[:450].rsplit(" ", 1)[0] + "…"
        return f"По данным базы знаний ({c.label}): {text}"
    return (
        "В базе знаний не найдено документов по этому вопросу, поэтому ответить по документам нельзя. "
        "Уточните вопрос или обратитесь к юристу."
    )


def _search(index: Index, query: str, prep: Prepared, kinds: set[str]) -> list[int]:
    return index.hybrid_multi([(query, prep.qvec)] + (prep.extra or []), kinds)


def weak_search(kb: KB, index: Index, prep: Prepared) -> bool:
    if prep.qvec is not None and index.vecs is not None:
        best = max((sc for _, sc in index.dense(prep.qvec)[:1]), default=0.0)
        return best < REWRITE_COS
    best = max((sc for _, sc in index.bm25(expand_query(search_query(prep)))[:1]), default=0.0)
    return best < REWRITE_BM25


def _parse_rewrite(text: str) -> tuple[str, list[str]] | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    q = data.get("question") if isinstance(data, dict) else None
    qs = data.get("queries") if isinstance(data, dict) else None
    if not isinstance(q, str) or not q.strip():
        return None
    qs = [x.strip()[:300] for x in qs if isinstance(x, str) and x.strip()][:3] if isinstance(qs, list) else []
    return q.strip()[:600], qs


def maybe_rewrite(kb: KB, index: Index, prep: Prepared, level: int, info: dict | None = None) -> None:
    if prep.rewrite_done:
        return
    f = runtime.FLAGS
    if level < 4 or f.compact or f.minimal or f.tiny:
        return
    if not prep.history and not weak_search(kb, index, prep):
        return
    with runtime.PROG.lock:
        left = max(1, runtime.PROG.n - runtime.PROG.finished)
    extra = runtime.TIMING.get("rewrite", runtime.TIMING.get("chat_full", 8.0))
    need = left * (runtime.question_cost(4) + extra) / max(1, MAX_WORKERS) * 1.2
    measured = MODE == "web" or "chat_full" in runtime.TIMING.ema
    if not measured or remaining() < 120 or need > remaining() - 15:
        f.inc("rewrite_skipped")
        return
    prep.rewrite_done = True
    t0 = time.monotonic()
    try:
        raw = chat(
            [
                {"role": "system", "content": SYSTEM_REWRITE},
                {"role": "user", "content": rewrite_message(prep.current, prep.history)},
            ],
            max_tokens=300,
            timeout=30.0,
        )
    except APIError as exc:
        f.inc("rewrite_failed")
        log(f"Переформулирование не удалось: {exc}")
        return
    runtime.TIMING.add("rewrite", time.monotonic() - t0)
    parsed = _parse_rewrite(raw)
    if not parsed:
        f.inc("rewrite_failed")
        return
    standalone, queries = parsed
    queries = [standalone] + [q for q in queries if q != standalone]
    vecs: list = [None] * len(queries)
    if index.vecs is not None and prep.qvec is not None:
        try:
            vecs = embed(queries)
        except APIError:
            vecs = [None] * len(queries)
    prep.standalone = standalone
    prep.extra = list(zip(queries, vecs))
    f.inc("rewrite")
    if info is not None:
        info["rewrite"] = {"standalone": standalone, "queries": queries[1:], "sec": round(time.monotonic() - t0, 2)}


def select_law(
    kb: KB,
    index: Index,
    prep: Prepared,
    use_rerank: bool,
    budget_chars: int = LAW_BUDGET_CHARS,
    diag: dict | None = None,
):
    query = search_query(prep) if prep.history else prep.current
    ranked = _search(index, query, prep, {"law"})[:CANDIDATES]
    if not ranked:
        return [], []
    budget = _law_budget(kb, index, query, budget_chars)
    gp_room = (budget_chars - budget) if GENERAL_POINTS else 0
    order = _rerank_law(kb, prep, query, ranked) if use_rerank else ranked
    items: dict[int, list[Chunk] | None] = {}
    reasons: dict[int, str] = {}
    budget = _add_found(kb, order, items, reasons, budget)
    budget = _add_general_points(kb, order, items, reasons, budget, gp_room, query)
    xref: dict[int, str] = {}
    absent: list[tuple[str, str]] = []
    if CROSS_REFS:
        budget = _add_cross_refs(kb, order, items, reasons, budget, xref, absent)
    rank_of = {}
    for r, ci in enumerate(order):
        rank_of.setdefault(kb.chunks[ci].doc, r)
    law_items = sorted(items.items(), key=lambda x: rank_of.get(x[0], 999))
    if diag is not None:
        diag.update(
            law_hybrid=list(ranked[:40]),
            law_order=list(order[:40]),
            rerank_used=order is not ranked,
            law_reason=reasons,
        )
        if CROSS_REFS:
            diag.update(law_xref=xref, law_absent=absent)
    return law_items, order


def _law_budget(kb: KB, index: Index, query: str, budget_chars: int) -> int:
    budget = budget_chars
    allbm = index.bm25(expand_query(query))
    if allbm:
        best_law = max((sc for ci, sc in allbm if kb.docs[kb.chunks[ci].doc].kind == "law"), default=0.0)
        best_comp = max((sc for ci, sc in allbm if kb.docs[kb.chunks[ci].doc].kind == "company"), default=0.0)
        top3_law = any(kb.docs[kb.chunks[ci].doc].kind == "law" for ci, _ in allbm[:3])
        if best_comp > 0 and best_law < 0.6 * best_comp and not top3_law:
            budget = min(budget, 5000)
    return budget


def _rerank_law(kb: KB, prep: Prepared, query: str, ranked: list[int]) -> list[int]:
    rr = rerank(
        prep.standalone or (prep.current if not prep.history else query[:2000]),
        [kb.chunks[i].search[:1200] for i in ranked],
        top_n=len(ranked),
    )
    if not rr:
        return ranked
    rr_sorted = [ranked[i] for i, _ in sorted(rr, key=lambda x: -x[1])]
    return rr_sorted + [i for i in ranked if i not in set(rr_sorted)]


def _add_found(kb: KB, order: list[int], items: dict, reasons: dict, budget: int) -> int:
    whole = []
    for ci in order[:4]:
        d = kb.chunks[ci].doc
        if len(kb.docs[d].body) <= min(SMALL_LAW_CHARS, budget) and d not in whole:
            whole.append(d)
    for d in whole:
        size = len(kb.docs[d].body)
        if size <= budget:
            items[d] = None
            budget -= size
    for ci in order[:LAW_TOP]:
        c = kb.chunks[ci]
        if c.doc in items and items[c.doc] is None:
            continue
        if len(c.text) > budget:
            continue
        items.setdefault(c.doc, [])
        if c not in items[c.doc]:
            items[c.doc].append(c)
            budget -= len(c.text)
            reasons[c.idx] = "search"
    return budget


def _add_general_points(
    kb: KB, order: list[int], items: dict, reasons: dict, budget: int, gp_room: int, query: str
) -> int:
    seen_docs = []
    for ci in order[:3]:
        d = kb.chunks[ci].doc
        if d in seen_docs or (d in items and items[d] is None):
            continue
        seen_docs.append(d)
        extra = 0
        room = gp_room if gp_room and _topic_hit(kb.docs[d], query) else 0
        for c in kb.chunks:
            if c.doc != d or not c.point or c.sub or len(c.text) > 700 or "(продолжение" in c.text:
                continue
            if c in (items.get(d) or []) or len(c.text) > budget + room or extra + len(c.text) > 2500:
                continue
            items.setdefault(d, [])
            items[d].append(c)
            if len(c.text) > budget:
                room -= len(c.text) - budget
                gp_room -= len(c.text) - budget
                budget = 0
            else:
                budget -= len(c.text)
            extra += len(c.text)
            reasons[c.idx] = "general_point"
    return budget


_TOPIC_GENERIC = {
    "налог",
    "налогов",
    "ставк",
    "глав",
    "кодекс",
    "стат",
    "закон",
    "федеральн",
    "российск",
    "федерац",
    "организац",
}


def _topic_hit(d: Doc, query: str) -> bool:
    title = norm((d.search_extra or d.title).split("«", 1)[-1])
    stems = {STEMMER.stem(w) for w in WORD_RE.findall(title) if not w.isdigit() and len(w) > 2 and w not in STOPWORDS}
    q = {STEMMER.stem(w) for w in WORD_RE.findall(norm(expand_query(query))) if len(w) > 2 and w not in STOPWORDS}
    return bool((stems - _TOPIC_GENERIC) & q)


XREF_MAX = 3
XREF_ABSENT_MAX = 5
XREF_SOURCES = 5


def _add_cross_refs(kb: KB, order: list[int], items: dict, reasons: dict, budget: int, xref: dict, absent: list) -> int:
    refs = cross_refs(kb)
    in_ctx = {c.idx for chs in items.values() if chs for c in chs}
    sources = [
        ci
        for ci in order[:LAW_TOP]
        if kb.chunks[ci].doc in items and (items[kb.chunks[ci].doc] is None or ci in in_ctx)
    ][:XREF_SOURCES]
    added = 0
    for ci in sources:
        src = kb.chunks[ci]
        for ref in refs.get(ci, []):
            for a in ref["absent"] if ci == order[0] else []:
                if a not in absent and len(absent) < XREF_ABSENT_MAX:
                    absent.append(a)
            for t in ref["targets"]:
                tc = kb.chunks[t]
                if added >= XREF_MAX or t in in_ctx or len(tc.text) > budget:
                    continue
                if tc.doc in items and items[tc.doc] is None:
                    continue
                items.setdefault(tc.doc, []).append(tc)
                in_ctx.add(t)
                budget -= len(tc.text)
                reasons[t] = "cross_ref"
                xref[t] = src.clause if tc.doc == src.doc and src.clause else src.label
                added += 1
    return budget


def company_context(
    kb: KB, index: Index, prep: Prepared, compact: bool = False, minimal: bool = False, diag: dict | None = None
) -> list[Doc]:
    docs = kb.company
    full = not compact and not minimal
    if full and not COMPANY_TOP and sum(len(d.text) for d in docs) <= COMPANY_FULL_LIMIT:
        if diag is not None:
            diag["company_reason"] = {d.idx: "company_full" for d in docs}
        return docs
    limit = 5000 if minimal else (10000 if compact else COMPANY_FULL_LIMIT)
    max_docs = 2 if minimal else (4 if compact else (COMPANY_TOP_DOCS if COMPANY_TOP else 8))
    ranked = _search(index, search_query(prep, assistant=True), prep, {"company"})
    chosen: list[Doc] = []
    size = 0
    for ci in ranked:
        d = kb.docs[kb.chunks[ci].doc]
        if d in chosen:
            continue
        group = [d] + [o for o in related_docs(kb, d) if o not in chosen]
        for g in group:
            if g in chosen or size + len(g.text) > limit or len(chosen) >= max_docs + 2:
                continue
            chosen.append(g)
            size += len(g.text)
            if diag is not None:
                diag.setdefault("company_reason", {})[g.idx] = "search" if g is d else "related"
        if len(chosen) >= max_docs:
            break
    if full and COMPANY_TOP and not chosen and sum(len(d.text) for d in docs) <= COMPANY_FULL_LIMIT:
        if diag is not None:
            diag["company_reason"] = {d.idx: "company_full" for d in docs}
        return docs
    return [d for d in docs if d in chosen]
