from __future__ import annotations

import re
from datetime import date
from typing import TYPE_CHECKING

from dates import event_periods
from kb import doc_clauses

if TYPE_CHECKING:
    from kb import KB, Doc


_COMPANY_MENTIONS = [
    (r"доп\w*\.?\s+соглашени\w*\s*(?:№\s*)?1\b", "Доп соглашение 1"),
    (r"ПТ-05(?:/2023)?|Альфа-Трейдинг", "ПТ-05-2023"),
    (r"А-08(?:/2024)?|Северн\w*\s+Склад", "А-08-2024"),
    (r"П-17(?:/2024)?|Альфа-Трейд(?!инг)", "П-17-2024"),
    (r"К-12\b|комплаенс", "К-12"),
    (r"ЮО-7\b|служебн\w*\s+записк", "Служебная записка"),
    (r"Регламент\w*", "Регламент"),
]
_LAW_MENTION = re.compile(
    r"(ТК|НК|КоАП)\s*РФ\W{0,3}(?:ст\.|стать\w*)\s*(\d+(?:\.\d+)?)|"
    r"(?:ст\.|стать\w*)\s*(\d+(?:\.\d+)?)\s*(ТК|НК|КоАП)\s*РФ|"
    r"(\d+-ФЗ)\W{0,3}(?:ст\.|стать\w*)\s*(\d+)|(?:ст\.|стать\w*)\s*(\d+)(?:,?\s*ч\.\s*\d+)?\s*"
    r"(?:Федеральн\w+\s+закона\s+)?(?:№\s*)?(\d+-ФЗ)",
    re.I,
)
CLAUSE_REF = re.compile(r"(?<![\w.])(пп?|ч|подп)\.\s*(\d+(?:\.\d+)*(?:\s*(?:,|и|–|-|—)\s*\d+(?:\.\d+)*)*)", re.I)


def expand_nums(raw: str) -> list[str]:
    out = []
    for part in re.split(r"\s*(?:,|\bи\b)\s*", raw):
        m = re.fullmatch(r"(\d+(?:\.\d+)*)\s*[–—-]\s*(\d+(?:\.\d+)*)", part.strip())
        if m:
            a, b = m.group(1).split("."), m.group(2).split(".")
            if len(a) == len(b) and a[:-1] == b[:-1] and int(a[-1]) <= int(b[-1]) <= int(a[-1]) + 20:
                out += [".".join(a[:-1] + [str(k)]) for k in range(int(a[-1]), int(b[-1]) + 1)]
            else:
                out += [m.group(1), m.group(2)]
        elif re.fullmatch(r"\d+(?:\.\d+)*", part.strip()):
            out.append(part.strip())
    return out


def mentions(kb: KB, text: str, qtext: str) -> list[tuple[int, int, Doc]]:
    raw: list[tuple[int, int, str, str | None]] = []
    for pat, key in _COMPANY_MENTIONS:
        for m in re.finditer(pat, text, re.I):
            qual = None
            if key == "Регламент":
                after = text[m.end() : m.end() + 70]
                if re.search(r"^\W*(?:\w+\s+){0,4}?(?:ред\w*\.?\s*(?:от\s*)?10\.01\.2023|2023|утрат)", after, re.I):
                    qual = "2023"
                elif re.search(
                    r"^\W*(?:\w+\s+){0,4}?(?:ред\w*\.?\s*(?:от\s*)?01\.03\.2025|2025|действующ)", after, re.I
                ):
                    qual = "2025"
            raw.append((m.start(), m.end(), key, qual))
    for m in re.finditer(r"(?<![\w-])договор\w*(?!\s+(?:поставки|аренды)?\s*№)", text, re.I):
        raw.append((m.start(), m.end(), "договор", None))
    raw.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    found: list[tuple[int, int, Doc]] = []
    periods = event_periods(qtext)
    last_contract = last_reg = None
    for s, e, key, qual in raw:
        if any(a <= s < b for a, b, _ in found):
            continue
        doc = None
        if key == "Регламент":
            regs = [d for d in kb.company if "Регламент" in d.label]
            if qual:
                doc = next((d for d in regs if qual in d.label), None)
            elif last_reg is not None:
                doc = last_reg
            else:
                fit = [d for d in regs if applies(d, periods)]
                doc = fit[0] if len(fit) == 1 else next((d for d in regs if "2025" in d.label), None)
            last_reg = doc or last_reg
        elif key == "договор":
            doc = last_contract
        else:
            doc = next((d for d in kb.company if key in d.label), None)
            if doc is not None and doc_type_is_contract(doc):
                last_contract = doc
        if doc is not None:
            found.append((s, e, doc))
    for m in _LAW_MENTION.finditer(text):
        g = m.groups()
        code, art = (g[0], g[1]) if g[0] else (g[3], g[2]) if g[3] else (g[4], g[5]) if g[4] else (g[7], g[6])
        code = code.upper().replace("КОАП", "КоАП")
        for d in kb.laws:
            if (
                code.endswith("ФЗ") and code in d.label and d.label.endswith(f"ст. {art}")
            ) or d.label == f"{code} РФ, ст. {art}":
                found.append((m.start(), m.end(), d))
                break
    return sorted(found, key=lambda x: x[0])


def doc_type_is_contract(d: Doc) -> bool:
    return d.kind == "company" and d.title.upper().startswith("ДОГОВОР")


def clause_key(d: Doc, kind: str, num: str, prev_point: str | None) -> str:
    if d.kind == "company":
        return num
    word = "п." if d.label.startswith("НК") else "ч."
    if kind.lower() in ("пп", "подп") and prev_point:
        return f"{word} {prev_point}, пп. {num}"
    return f"{word} {num}"


def quote_span(d: Doc, start: int, end: int, limit: int = 600) -> tuple[int, int]:
    if end - start > limit:
        cut = d.text.rfind(" ", start, start + limit)
        end = cut if cut > start + limit // 2 else start + limit
    return start, end


def applies(d: Doc, periods: list) -> bool | None:
    if not periods or not (d.valid_from or d.valid_to):
        return None
    vf = d.valid_from or date(1900, 1, 1)
    vt = d.valid_to or date(2999, 12, 31)
    return all(vf <= end and start <= vt for start, end, _ in periods)


def kb_clauses(d: Doc) -> list[dict]:
    if getattr(d, "_clauses", None) is None:
        d._clauses = doc_clauses(d)
    return d._clauses


_CODE_RE = re.compile(
    r"№\s*([A-Za-zА-ЯЁа-яё0-9][\w\-/.]*\d[\w\-/]*)|(?<![\w-])([А-ЯЁA-Z]{1,4}-\d+(?:/\d+)?)(?![\w-])|"
    r"(?<![\w-])(\d+-ФЗ)\b"
)
_ART_REF = re.compile(
    r"(ТК|НК|КоАП)\s*РФ\W{0,3}(?:ст\.|стать\w*)\s*(\d+(?:\.\d+)?)|"
    r"(?:ст\.|стать\w*)\s*(\d+(?:\.\d+)?)\s*(?:,\s*ч\.\s*\d+\s*)?(ТК|НК|КоАП)\s*РФ",
    re.I,
)
_OTHER_ART = re.compile(
    r"^\s*(?:,\s*)?(?:(?:ч|п)\.\s*\d+(?:[\s,–-]+\d+)*\s*)?(?:ст\.|стать\w*)\s*(\d+(?:\.\d+)?)", re.I
)


def _kb_norm(kb: KB) -> str:
    if getattr(kb, "_norm_text", None) is None:
        kb._norm_text = "\n".join(d.text for d in kb.docs).lower().replace("ё", "е").replace("/", "-")
    return kb._norm_text


def _article_known(kb: KB, num: str) -> bool:
    pat = re.compile(rf"(?:стать\w*|ст\.)\s*{re.escape(num)}(?![\d.])", re.I)
    return any(d.label.endswith(f"ст. {num}") or pat.search(d.text) for d in kb.docs)


def unverified_refs(kb: KB, answer: str, qtext: str, context_docs: list) -> list[tuple[str, str]]:
    bad: list[tuple[str, str]] = []
    found = mentions(kb, answer, qtext)
    for m in CLAUSE_REF.finditer(answer):
        nxt = _OTHER_ART.match(answer[m.end() : m.end() + 40])
        if nxt or re.match(r"\s*(?:ч\.\s*\d+\s*)?(?:ст\.|стать)", answer[m.end() : m.end() + 12], re.I):
            art = nxt.group(1) if nxt else None
            if art and not _article_known(kb, art):
                bad.append((answer[m.start() : m.end() + nxt.end()].strip(), "ref"))
            continue
        before = [x for x in found if x[0] <= m.start()]
        after = [x for x in found if x[0] >= m.end() and x[0] - m.end() < 60]
        if after and re.fullmatch(r"\s{0,3}", answer[m.end() : after[0][0]]):
            cands = [after[0][2]]
        elif before and answer.rfind("(", 0, m.start()) > before[-1][1] and after:
            cands = [after[0][2]]
        elif before:
            cands = [before[-1][2]]
        elif after:
            cands = [after[0][2]]
        else:
            cands = []
        cands += [x[2] for x in reversed(before)] + [x[2] for x in after]
        pool = cands or list(context_docs)
        nums = expand_nums(m.group(2))
        missing = []
        for num in nums:
            ok = False
            for d in pool or context_docs:
                clauses = kb_clauses(d)
                if d.kind == "law" and not clauses:
                    ok = True
                    break
                key = clause_key(d, m.group(1), num, None)
                if any(c["clause"] == key or c["clause"].startswith(key + ",") for c in clauses):
                    ok = True
                    break
                if m.group(1).lower() in ("пп", "подп") and any(c["clause"].endswith(f"пп. {num}") for c in clauses):
                    ok = True
                    break
            if not ok:
                missing.append(num)
        if missing:
            bad.append((m.group(0).strip(), "ref"))
    norm_kb = _kb_norm(kb)
    for m in _CODE_RE.finditer(answer):
        code = next(g for g in m.groups() if g)
        c = code.lower().replace("ё", "е").replace("/", "-").rstrip(".-")
        if c.isdigit() and len(c) <= 2:
            continue
        if c not in norm_kb:
            bad.append((code, "ref"))
    for m in _ART_REF.finditer(answer):
        num = m.group(2) or m.group(3)
        code = (m.group(1) or m.group(4)).upper().replace("КОАП", "КоАП")
        if not any(d.label == f"{code} РФ, ст. {num}" for d in kb.docs) and not _article_known(kb, num):
            bad.append((m.group(0).strip(), "ref"))
    seen, out = set(), []
    for sp, kind in bad:
        if sp not in seen:
            seen.add(sp)
            out.append((sp, kind))
    return out
