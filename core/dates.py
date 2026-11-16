from __future__ import annotations

import re
from datetime import date, timedelta
from typing import TYPE_CHECKING

from textproc import norm, tokenize

if TYPE_CHECKING:
    from kb import KB


DATE_NUM = re.compile(r"(?<!\d)(\d{1,2})\.(\d{1,2})\.(\d{4})(?!\d)")


def mkdate(d: str, m: str, y: str) -> date | None:
    try:
        return date(int(y), int(m), int(d))
    except ValueError:
        return None


def fmt_date(d: date | None) -> str:
    return d.strftime("%d.%m.%Y") if d else ""


MONTHS = [
    ("январ", 1),
    ("феврал", 2),
    ("март", 3),
    ("апрел", 4),
    ("ма", 5),
    ("июн", 6),
    ("июл", 7),
    ("август", 8),
    ("сентябр", 9),
    ("октябр", 10),
    ("ноябр", 11),
    ("декабр", 12),
]
MONTH_WORD = (
    r"(январ\w*|феврал\w*|март\w*|апрел\w*|ма[йяе]|июн\w*|июл\w*|август\w*|сентябр\w*|октябр\w*|ноябр\w*|декабр\w*)"
)
DMY_TEXT = re.compile(r"(?<!\d)(\d{1,2})\s+" + MONTH_WORD + r"\s+(\d{4})", re.I)
MY_TEXT = re.compile(r"(?<![\d\w])" + MONTH_WORD + r"\s+(\d{4})", re.I)
YEAR_TEXT = re.compile(
    r"(?:\b(?:в|во|за|с|до|на|по|конец|начало|начале|конце)\s+(20\d{2})\b(?![/\d-])|"
    r"(?<![\d/-])(20\d{2})\s*(?:год\w*|г\.))",
    re.I,
)


def month_no(word: str) -> int:
    w = norm(word)
    for stem, n in MONTHS:
        if w.startswith(stem):
            return n
    return 0


def event_periods(text: str) -> list[tuple[date, date, str]]:
    out: list[tuple[date, date, str]] = []
    used: list[tuple[int, int]] = []

    def free(span: tuple[int, int]) -> bool:
        return all(span[1] <= a or span[0] >= b for a, b in used)

    for m in DATE_NUM.finditer(text):
        d = mkdate(*m.groups())
        if d and 1990 < d.year < 2100:
            out.append((d, d, m.group(0)))
            used.append(m.span())
    for m in DMY_TEXT.finditer(text):
        if free(m.span()):
            mo = month_no(m.group(2))
            d = mkdate(m.group(1), str(mo), m.group(3)) if mo else None
            if d:
                out.append((d, d, m.group(0)))
                used.append(m.span())
    for m in MY_TEXT.finditer(text):
        if free(m.span()):
            mo, y = month_no(m.group(1)), int(m.group(2))
            if mo and 1990 < y < 2100:
                start = date(y, mo, 1)
                end = date(y + (mo == 12), mo % 12 + 1, 1) - timedelta(days=1)
                out.append((start, end, m.group(0)))
                used.append(m.span())
    for m in YEAR_TEXT.finditer(text):
        if free(m.span()):
            y = int(m.group(1) or m.group(2))
            out.append((date(y, 1, 1), date(y, 12, 31), m.group(0).strip()))
            used.append(m.span())
    return out


def date_note(question: str, kb: KB) -> str:
    periods = event_periods(question)
    dated = [d for d in kb.company if d.valid_from or d.valid_to]
    if not periods or not dated:
        return ""
    lines = []
    for start, end, raw in periods[:3]:
        ok, no = [], []
        for d in dated:
            vf = d.valid_from or date(1900, 1, 1)
            vt = d.valid_to or date(2999, 12, 31)
            name = f"«{d.label}»"
            if vf <= end and start <= vt:
                part = "" if (vf <= start and end <= vt) else " (действовал лишь часть этого периода)"
                ok.append(name + part)
            else:
                no.append(name)
        line = f"Период из вопроса «{raw}»: применимы {', '.join(ok) if ok else 'нет документов с датами действия'}"
        if no:
            line += f"; не действовали в этот период: {', '.join(no)}"
        lines.append(line + ".")
    return "\n".join(lines)


_LAW_CHANGE = re.compile(
    r"^((?:Пункт|Подпункт|Часть|Статья|Абзац)\s+[\w.\-]+)[^,]*?\s+(?:изменен\w*|изложен\w*|дополнен\w*)"
    r"\s+с\s+(\d{1,2})\s+" + MONTH_WORD + r"\s+(\d{4})",
    re.I,
)


def _overlap(start: date, end: date, d: date) -> str:
    if end < d:
        return "before"
    if start >= d:
        return "after"
    return "spans"


_GENERIC = {
    "договор",
    "подписа",
    "подписан",
    "срок",
    "нужн",
    "ли",
    "год",
    "месяц",
    "ооо",
    "обществ",
    "документ",
    "пункт",
    "требован",
    "услов",
}


def _stems(text: str) -> set:
    return {t for t in tokenize(text) if not t.isdigit() and len(t) > 2 and t not in _GENERIC}


def _relevant(question: str, text: str, need: int = 2) -> bool:
    return len(_stems(question) & _stems(text)) >= need


def date_conditions(question: str, kb: KB, law_items: list) -> str:
    periods = event_periods(question)
    if not periods:
        return ""
    lines: list[str] = []
    qnorm = norm(question).replace("/", "-")
    for d in kb.company:
        for a in d.amendments:
            codes = re.findall(r"[а-яa-z]{1,4}-\d+(?:-\d+)?", norm(a.get("target", "")).replace("/", "-"))
            if not (
                _relevant(question, a["condition"] + " " + a.get("target", "") + " " + d.label)
                or any(c in qnorm for c in codes)
            ):
                continue
            for start, end, raw in periods[:3]:
                pos = _overlap(start, end, a["date"])
                when = fmt_date(a["date"])
                tgt = f" ({a['target']})" if a.get("target") else ""
                if a["kind"] == "from":
                    verdict = {
                        "before": f"в период «{raw}» ещё не применялось",
                        "after": f"в период «{raw}» применялось",
                        "spans": f"начало применяться внутри периода «{raw}» — важно, до или после {when}",
                    }[pos]
                    lines.append(
                        f"«{d.label}»{tgt}: требование действует с {when} — {verdict}. Условие: {a['condition']}"
                    )
                else:
                    verdict = {
                        "before": f"к событиям периода «{raw}» не применяется",
                        "after": f"к событиям периода «{raw}» применяется",
                        "spans": f"в периоде «{raw}» зависит от точной даты (до или после {when})",
                    }[pos]
                    lines.append(f"«{d.label}»{tgt}: применяется только при условии — {a['condition']} — {verdict}.")
    seen = set()
    law_lines: list[str] = []
    for d, chs in law_items:
        if not _relevant(question, (d.search_extra or d.title), need=1):
            continue
        texts = [d.body] if chs is None else [c.text for c in chs]
        notes = list(d.art_notes) + [
            ln.strip()[len("[Пометка редакции: ") : -1]
            for t in texts
            for ln in t.split("\n")
            if ln.strip().startswith("[Пометка редакции:")
        ]
        try:
            exported = date(*reversed([int(x) for x in d.export_date.split(".")])) if d.export_date else None
        except ValueError:
            exported = None
        for n in notes:
            m = _LAW_CHANGE.match(n)
            if not m:
                continue
            mo = month_no(m.group(3))
            changed = mkdate(m.group(2), str(mo), m.group(4)) if mo else None
            if not changed or (exported and changed > exported):
                continue
            for start, end, raw in periods[:3]:
                key = (d.idx, m.group(1).lower(), raw)
                if end < changed and key not in seen:
                    seen.add(key)
                    law_lines.append(
                        f"{d.label}, {m.group(1).lower()}: изменения с {fmt_date(changed)}; в базе только "
                        f"действующая редакция — на дату события «{raw}» могла действовать иная, её текста в базе нет."
                    )
    return "\n".join(lines + law_lines[:3])
