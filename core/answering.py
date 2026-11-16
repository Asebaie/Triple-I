from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

import runtime
from checks import clean_answer, drop_ref_sentences, drop_sentences, trim_after_source, unverified
from config import (
    ANSWER_TEMPLATE,
    CHECK_FIXES,
    CHECK_REFS,
    CONCISE,
    DATE_CONDITIONS,
    REWRITE,
    REWRITE_PROMPT,
    SOFT_DEGRADE,
    debug_dump,
    log,
    remaining,
)
from dates import date_conditions, date_note
from dialog import Prepared, search_query, user_text
from llm import APIError, chat
from prompts import (
    SYSTEM_TEXT,
    SYSTEM_TEXT_MINI,
    SYSTEM_TEXT_SHORT,
    SYSTEM_TINY,
    build_user_message,
    repair_numbers_message,
    repair_refs_message,
)
from refs import unverified_refs
from retrieval import company_context, maybe_rewrite, select_law
from runtime import per_question_budget

if TYPE_CHECKING:
    from index import Index
    from kb import KB


def answer_tiny(kb: KB, index: Index, prep: Prepared, trace=None, info: dict | None = None) -> str:
    found = index.hybrid(search_query(prep), prep.qvec, {"company", "law"})[:6]
    parts, size = [], 0
    for ci in found:
        c = kb.chunks[ci]
        if size + len(c.text) > 2800:
            continue
        parts.append(f"[{c.label}]\n{c.text}")
        size += len(c.text)
    hist = "\n".join(("Пользователь: " if r == "user" else "Ассистент: ") + t for r, t in prep.history)
    user = (
        "Фрагменты:\n"
        + "\n\n".join(parts)
        + (f"\n\nИстория диалога:\n{hist}" if hist else "")
        + f"\n\nВопрос:\n{prep.current}"
    )
    if trace:
        trace("generate")
    try:
        ans = clean_answer(chat([{"role": "system", "content": SYSTEM_TINY}, {"role": "user", "content": user}]))
    except APIError as exc:
        if not exc.transient:
            with runtime.FLAGS.lock:
                runtime.FLAGS.chat_fail_streak += 1
        raise
    if not ans:
        raise APIError("пустой ответ после очистки", transient=True)
    runtime.FLAGS.inc("level0")
    if info is not None:
        info.update(level=0, chunks=list(found))
    return ans


_SIZE_ERROR = re.compile(r"(?i)context|too long|too large|maximum|max_?tokens|token limit|length|слишком|превыш")


def _size_error(exc: APIError) -> bool:
    return exc.status == 413 or bool(_SIZE_ERROR.search(exc.detail or ""))


def _maybe_recover() -> None:
    f = runtime.FLAGS
    with f.lock:
        if f.degraded_at >= 0 and (f.compact or f.minimal or f.tiny):
            with runtime.PROG.lock:
                done = runtime.PROG.finished
            if done - f.degraded_at >= 10:
                f.compact = f.minimal = f.tiny = False
                f.degraded_at = -1
                log("Пробуем снова полный режим запроса")


def _soft_4xx(
    kb: KB, index: Index, qtext: str, prep: Prepared, level: int, exc: APIError, trace, info, retried: bool
) -> str:
    if not retried:
        log(f"Прокси отклонил запрос ({exc}) — повторяем тот же уровень")
        time.sleep(1.0)
        return answer_one(kb, index, qtext, prep, level, trace, info, _retry=True)
    f = runtime.FLAGS
    with f.lock:
        if not prep.soft4xx_counted:
            prep.soft4xx_counted = True
            f.soft4xx_streak += 1
        if f.soft4xx_streak >= 3:
            f.soft4xx_streak = 0
            if level >= 3:
                f.compact = True
            elif level == 2:
                f.minimal = True
            else:
                f.tiny = True
            with runtime.PROG.lock:
                f.degraded_at = runtime.PROG.finished
            log(f"Три вопроса подряд с отказом прокси ({exc}) — временно понижаем режим")
    f.inc("soft4xx")
    if level >= 3:
        return answer_one(kb, index, qtext, prep, 2, trace, info)
    if level == 2:
        return answer_one(kb, index, qtext, prep, 1, trace, info)
    return answer_tiny(kb, index, prep, trace, info)


def answer_one(
    kb: KB,
    index: Index,
    qtext: str,
    prep: Prepared,
    level: int,
    trace=None,
    info: dict | None = None,
    _retry: bool = False,
) -> str:
    if trace:
        trace("search")
    if SOFT_DEGRADE:
        _maybe_recover()
    if runtime.FLAGS.tiny:
        return answer_tiny(kb, index, prep, trace, info)
    if runtime.FLAGS.minimal:
        level = 1
    elif runtime.FLAGS.compact:
        level = min(level, 2)
    sel: dict = {}
    law_items, comp, dnote, user_msg = _prepare_message(kb, index, qtext, prep, level, sel)
    system = SYSTEM_TEXT if level >= 3 else (SYSTEM_TEXT_SHORT if level == 2 else SYSTEM_TEXT_MINI)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
    if trace:
        trace("generate")
    t0 = time.monotonic()
    try:
        ans = clean_answer(chat(messages))
    except APIError as exc:
        lower = _on_reject(kb, index, qtext, prep, level, exc, trace, info, _retry)
        if lower is None:
            raise
        return lower
    runtime.TIMING.add(
        {4: "chat_full", 3: "chat_full", 2: "chat_compact"}.get(level, "chat_min"), time.monotonic() - t0
    )
    if SOFT_DEGRADE and level >= 3:
        with runtime.FLAGS.lock:
            runtime.FLAGS.soft4xx_streak = 0
    if not ans:
        raise APIError("пустой ответ после очистки", transient=True)
    draft = ans
    if trace:
        trace("verify")
    ctx_docs = list(comp) + [kb.docs[d] for d, _ in law_items]
    ans, bad, refs = _verify_and_repair(kb, ans, messages, user_msg, qtext, ctx_docs, level)
    ans = _trim_answer(ans, bad, refs)
    runtime.FLAGS.inc(f"level{level}")
    record = {
        "question": qtext,
        "current": prep.current,
        "level": level,
        "laws": [(kb.docs[d].label, "целиком" if chs is None else [c.label for c in chs]) for d, chs in law_items],
        "company": [d.label for d in comp],
        "date_note": dnote,
        "draft": draft,
        "unverified": bad,
        "final": ans,
        "prompt_chars": len(user_msg),
        **sel,
        "law_chunks": [c.idx for _, chs in law_items if chs for c in chs],
        "law_whole": [d for d, chs in law_items if chs is None],
        "user_msg": user_msg,
        "law_items_idx": [[d, None if chs is None else [c.idx for c in chs]] for d, chs in law_items],
    }
    if CHECK_REFS:
        record["unverified_refs"] = refs
    debug_dump(record)
    if info is not None:
        info.update(record, law_doc_idx=[d for d, _ in law_items], company_doc_idx=[d.idx for d in comp])
    return ans


def _prepare_message(kb: KB, index: Index, qtext: str, prep: Prepared, level: int, sel: dict):
    if REWRITE:
        maybe_rewrite(kb, index, prep, level, sel)
    if level >= 3:
        law_items, comp = [(d.idx, None) for d in kb.laws], kb.company
    else:
        law_items, _ = select_law(
            kb, index, prep, use_rerank=False, budget_chars=4000 if level == 2 else 2000, diag=sel
        )
        comp = company_context(kb, index, prep, compact=True, minimal=level <= 1, diag=sel)
    date_text = user_text(prep) if CHECK_FIXES else qtext
    dnote = date_note(date_text, kb)
    if DATE_CONDITIONS:
        extra = date_conditions(date_text, kb, [(kb.docs[d], chs) for d, chs in law_items])
        dnote = "\n".join(x for x in (dnote, extra) if x)
    user_msg = build_user_message(
        kb,
        comp,
        law_items,
        dnote,
        prep.current,
        prep.history,
        xref=sel.get("law_xref"),
        absent=sel.get("law_absent"),
        standalone=prep.standalone if REWRITE and REWRITE_PROMPT else None,
    )
    return law_items, comp, dnote, user_msg


def _on_reject(
    kb: KB, index: Index, qtext: str, prep: Prepared, level: int, exc: APIError, trace, info, retried: bool
) -> str | None:
    if SOFT_DEGRADE and not exc.transient and not _size_error(exc):
        return _soft_4xx(kb, index, qtext, prep, level, exc, trace, info, retried)
    if not exc.transient and level >= 2:
        with runtime.FLAGS.lock:
            if level >= 3 and not runtime.FLAGS.compact:
                runtime.FLAGS.compact = True
                log(f"Прокси отклонил полный запрос ({exc}) — переходим на компактный режим")
            elif level == 2 and not runtime.FLAGS.minimal:
                runtime.FLAGS.minimal = True
                log(f"Прокси отклонил компактный запрос ({exc}) — переходим на минимальный режим")
        return answer_one(kb, index, qtext, prep, 2 if level >= 3 else 1, trace, info)
    if not exc.transient and level == 1:
        with runtime.FLAGS.lock:
            if not runtime.FLAGS.tiny:
                runtime.FLAGS.tiny = True
                log(f"Прокси отклонил минимальный запрос ({exc}) — переходим на короткий запрос")
        return answer_tiny(kb, index, prep, trace, info)
    return None


def _verify_and_repair(
    kb: KB, ans: str, messages: list[dict], user_msg: str, qtext: str, ctx_docs: list, level: int
) -> tuple[str, list, list]:
    bad = unverified(ans, user_msg, qtext)
    refs = [sp for sp, _ in unverified_refs(kb, ans, qtext, ctx_docs)] if CHECK_REFS else []
    full = runtime.TIMING.get("chat_full", 8.0)
    can_repair = level >= 4 and remaining() > 60 and per_question_budget() >= 2.2 * full
    if (bad or refs) and can_repair:
        spans = [sp for sp, _ in bad]
        if bad:
            log(f"Неподтверждённые числа {spans[:5]} — запрашиваю исправление")
            fix = repair_numbers_message(ans, spans, refs)
        else:
            log(f"Неподтверждённые ссылки {refs[:5]} — запрашиваю исправление")
            fix = repair_refs_message(ans, refs)
        try:
            fixed = clean_answer(
                chat(messages + [{"role": "assistant", "content": ans}, {"role": "user", "content": fix}])
            )
            if fixed:
                ans = fixed
                runtime.FLAGS.inc("repaired")
                bad = unverified(ans, user_msg, qtext)
                if CHECK_REFS:
                    refs = [sp for sp, _ in unverified_refs(kb, ans, qtext, ctx_docs)]
        except APIError as exc:
            log(f"Исправление не удалось: {exc}")
    return ans, bad, refs


def _trim_answer(ans: str, bad: list, refs: list) -> str:
    strict = [sp for sp, kind in bad if kind in ("term", "date")]
    if strict:
        trimmed = drop_sentences(ans, strict)
        if trimmed != ans:
            runtime.FLAGS.inc("trimmed")
            log(f"Удалены предложения с неподтверждёнными сроками/датами: {strict[:5]}")
            ans = trimmed
    if refs:
        trimmed = drop_ref_sentences(ans, refs)
        if trimmed != ans:
            runtime.FLAGS.inc("ref_trimmed")
            log(f"Удалены предложения с неподтверждёнными ссылками: {refs[:5]}")
            ans = trimmed
    if ANSWER_TEMPLATE and CONCISE:
        cut = trim_after_source(ans)
        if cut != ans:
            runtime.FLAGS.inc("template_trimmed")
            ans = cut
    return ans
