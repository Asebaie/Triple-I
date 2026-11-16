from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dates import event_periods
from dialog import Prepared, event_text, search_query
from kb import doc_markup
from refs import CLAUSE_REF, applies, clause_key, expand_nums, kb_clauses, quote_span
from refs import mentions as doc_mentions

if TYPE_CHECKING:
    from index import Index
    from kb import KB, Chunk, Doc


STATUSES = ("answer", "partial", "no_info", "premise_error", "clarify", "out_of_scope", "error")


@dataclass
class Result:
    text: str
    status: str = "answer"
    citations: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    timing_ms: dict = field(default_factory=dict)
    debug: dict = field(default_factory=dict)
    contract_version: str = ""
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "contract_version": self.contract_version,
            "status": self.status,
            "text": self.text,
            "citations": self.citations,
            "warnings": self.warnings,
            "context": self.context,
            "timing_ms": self.timing_ms,
            "debug": self.debug,
        }


def extract_citations(kb: KB, text: str, qtext: str, ids: dict[int, str]) -> list[dict]:
    mentions = doc_mentions(kb, text, qtext)
    periods = event_periods(event_text(qtext))
    cites: dict[tuple[int, str | None], dict] = {}

    def add(d: Doc, clause: str | None) -> None:
        key = (d.idx, clause)
        if key in cites:
            return
        found = True
        if clause is None:
            first = d.text.split("\n", 1)[0]
            s, e = 0, len(first)
            if d.kind == "law":
                m = re.search(r"(?m)^Статья\s+\d.*$", d.text)
                if m:
                    s, e = m.start(), m.end()
        else:
            hit = next((c for c in kb_clauses(d) if c["clause"] == clause), None)
            if hit is None:
                found = False
                s, e = 0, len(d.text.split("\n", 1)[0])
            else:
                s, e = quote_span(d, hit["start"], hit["end"])
        cites[key] = {
            "doc_id": ids[d.idx],
            "doc_title": d.title,
            "doc_kind": d.kind,
            "clause": clause,
            "quote": d.text[s:e],
            "start": s,
            "end": e,
            "found": found,
            "applies_to_event_date": applies(d, periods),
        }

    for s, e, d in mentions:
        add(d, None)
    last_point: dict[int, str] = {}
    for m in CLAUSE_REF.finditer(text):
        before = [x for x in mentions if x[0] <= m.start()]
        after = [x for x in mentions if x[0] >= m.end() and x[0] - m.end() < 60]
        if after and re.fullmatch(r"\s{0,3}", text[m.end() : after[0][0]]):
            target = after[0][2]
        elif before and text.rfind("(", 0, m.start()) > before[-1][1] and after:
            target = after[0][2]
        elif before:
            target = before[-1][2]
        elif after:
            target = after[0][2]
        else:
            continue
        others = [x[2] for x in reversed(before)] + [x[2] for x in after]
        for num in expand_nums(m.group(2)):
            doc = target
            key = clause_key(doc, m.group(1), num, last_point.get(doc.idx))
            if not any(c["clause"] == key for c in kb_clauses(doc)):
                for o in others:
                    k2 = clause_key(o, m.group(1), num, last_point.get(o.idx))
                    if any(c["clause"] == k2 for c in kb_clauses(o)):
                        doc, key = o, k2
                        break
            if doc.kind == "law" and m.group(1).lower() not in ("пп", "подп"):
                last_point[doc.idx] = num
            add(doc, key)
    out = list(cites.values())
    with_clause = {c["doc_id"] for c in out if c["clause"]}
    return [c for c in out if c["clause"] or c["doc_id"] not in with_clause]


_NO_INFO = re.compile(
    r"(?i)в базе( знаний)? нет|нет документов|не найден\w* (документ|сведени)|"
    r"ответить по документам нельзя|не могу ответить по документам"
)
_SCOPE = re.compile(r"(?i)юридическ\w+ ассистент|не относится к (юридическ|теме)|вне (моей )?компетенц")
_CLARIFY = re.compile(r"(?i)\bуточните\b|просьба уточнить|нужно уточнить, (о каком|какой)|какой именно")
_PREMISE = re.compile(r"(?i)\bа не\s+(?:[«\"]?\d)|не соответствует|неверн|ошибочн|не подтвержда|предпосылк")
_PARTIAL = re.compile(
    r"(?i)(в базе|в документах|в договоре|в предоставленной базе)[^.]{0,60}(нет|отсутству|не указ|не установ)|"
    r"не хватает|частичн|отсутствуют в базе"
)


def classify_status(text: str, citations: list[dict]) -> str:
    has_cite = any(c["found"] for c in citations)
    if not has_cite and _SCOPE.search(text) and not _NO_INFO.search(text):
        return "out_of_scope"
    if not has_cite and _NO_INFO.search(text):
        return "no_info"
    if _CLARIFY.search(text):
        return "clarify"
    if _PREMISE.search(text) and has_cite:
        return "premise_error"
    if has_cite and (_PARTIAL.search(text) or _NO_INFO.search(text)):
        return "partial"
    if not has_cite and _SCOPE.search(text):
        return "out_of_scope"
    return "answer"


def build_warnings(kb: KB, qtext: str, info: dict, citations: list[dict], ids: dict[int, str]) -> list[dict]:
    out = []
    if info.get("date_note"):
        out.append({"code": "edition_note", "message": info["date_note"]})
    by_id = {v: k for k, v in ids.items()}
    cited = {by_id[c["doc_id"]] for c in citations}
    for di in sorted(cited):
        d = kb.docs[di]
        if d.removed:
            out.append(
                {
                    "code": "removed_instructions",
                    "doc_id": ids[di],
                    "message": f"В документе «{d.label}» удалена встроенная инструкция для ИИ: она не является "
                    "положением документа и не учитывалась",
                }
            )
    if event_periods(event_text(qtext)):
        for di in sorted(cited):
            d = kb.docs[di]
            if d.kind == "law":
                out.append(
                    {
                        "code": "law_current_edition_only",
                        "doc_id": ids[di],
                        "message": f"В базе только действующая редакция ({d.label}); на дату события могла "
                        "действовать иная",
                    }
                )
    amounts = [sp for sp, kind in info.get("unverified", []) if kind == "amount"]
    if amounts:
        out.append(
            {"code": "unverified_numbers", "message": "Числа не найдены в документах: " + ", ".join(amounts[:8])}
        )
    missing = [c for c in citations if not c["found"]]
    for c in missing:
        out.append(
            {"code": "clause_not_found", "doc_id": c["doc_id"], "message": f"Пункт {c['clause']} не найден в документе"}
        )
    return out


def _frag(kb: KB, c: Chunk, rank: int | None, reason: str) -> dict:
    return {
        "clause": c.clause,
        "start": c.start,
        "end": c.end,
        "rank": rank,
        "reason": reason,
        "edition_note": "[Пометка редакции:" in c.text,
    }


def build_context(
    kb: KB, index: Index, info: dict, prep: Prepared, qtext: str, ids: dict[int, str], citations: list[dict]
) -> dict:
    periods = event_periods(event_text(qtext))
    cited = {c["doc_id"] for c in citations}
    order = info.get("law_order") or []
    rank = {ci: i + 1 for i, ci in enumerate(order)}
    reasons = {int(k): v for k, v in (info.get("law_reason") or {}).items()}
    out: list[dict] = []

    def doc_entry(d: Doc, included: str, reason: str) -> dict:
        return {
            "doc_id": ids[d.idx],
            "doc_title": d.title,
            "doc_kind": d.kind,
            "included": included,
            "reason": reason,
            "applies_to_event_date": applies(d, periods),
            "cited": ids[d.idx] in cited,
        }

    if info.get("level") == 0:
        by_doc: dict[int, list] = {}
        for i, ci in enumerate(info.get("chunks") or []):
            by_doc.setdefault(kb.chunks[ci].doc, []).append(_frag(kb, kb.chunks[ci], i + 1, "search"))
        for di, frs in by_doc.items():
            out.append({**doc_entry(kb.docs[di], "fragments", "search"), "fragments": frs})
    else:
        hits: dict[int, list] = {}
        for r, ci in enumerate(index.hybrid(search_query(prep, assistant=True), prep.qvec, {"company"})[:10], 1):
            c = kb.chunks[ci]
            hits.setdefault(c.doc, []).append({"clause": c.clause, "start": c.start, "end": c.end, "rank": r})
        creason = {int(k): v for k, v in (info.get("company_reason") or {}).items()}
        for di in info.get("company_doc_idx") or []:
            out.append(
                {**doc_entry(kb.docs[di], "whole", creason.get(di, "company_full")), "search_hits": hits.get(di, [])}
            )
        for di, chs in info.get("law_items_idx") or []:
            d = kb.docs[di]
            if chs is None:
                sh = [
                    {
                        "clause": kb.chunks[ci].clause,
                        "start": kb.chunks[ci].start,
                        "end": kb.chunks[ci].end,
                        "rank": rank[ci],
                    }
                    for ci in order
                    if kb.chunks[ci].doc == di
                ]
                out.append({**doc_entry(d, "whole", "short_article"), "search_hits": sh})
            else:
                frs = [_frag(kb, kb.chunks[ci], rank.get(ci), reasons.get(ci, "search")) for ci in chs]
                main = next(
                    (r for r in ("search", "general_point", "cross_ref") if any(f["reason"] == r for f in frs)),
                    "search",
                )
                out.append({**doc_entry(d, "fragments", main), "fragments": frs})
    removed = []
    for e in out:
        d = kb.docs[next(k for k, v in ids.items() if v == e["doc_id"])]
        for r in doc_markup(d)["removed_instructions"]:
            removed.append({"doc_id": e["doc_id"], "start": r["start"], "end": r["end"], "note": r["note"]})
    return {"documents": out, "date_note": info.get("date_note") or "", "removed_from_context": removed}
