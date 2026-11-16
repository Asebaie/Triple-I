from __future__ import annotations

import re

from config import ANSWER_MAX_CHARS, CHECK_FIXES, FIX_DOCNUM
from dates import DATE_NUM, MONTH_WORD, month_no
from textproc import THINK_RE, norm

NUM_WORDS: dict[str, tuple[str, float]] = {}


def _reg(kind: str, value: float, forms: str) -> None:
    for f in forms.split():
        NUM_WORDS[f] = (kind, value)


for _v, _f in [
    (1, "один одна одно одного одной одному одним одних одну"),
    (2, "два две двух двум двумя"),
    (3, "три трех трем тремя"),
    (4, "четыре четырех четырем четырьмя"),
    (5, "пять пяти пятью"),
    (6, "шесть шести шестью"),
    (7, "семь семи семью"),
    (8, "восемь восьми восемью"),
    (9, "девять девяти девятью"),
]:
    _reg("u", _v, _f)
for _v, _f in [
    (10, "десять десяти десятью"),
    (11, "одиннадцать одиннадцати"),
    (12, "двенадцать двенадцати"),
    (13, "тринадцать тринадцати"),
    (14, "четырнадцать четырнадцати"),
    (15, "пятнадцать пятнадцати"),
    (16, "шестнадцать шестнадцати"),
    (17, "семнадцать семнадцати"),
    (18, "восемнадцать восемнадцати"),
    (19, "девятнадцать девятнадцати"),
    (20, "двадцать двадцати"),
    (30, "тридцать тридцати"),
    (40, "сорок сорока"),
    (50, "пятьдесят пятидесяти"),
    (60, "шестьдесят шестидесяти"),
    (70, "семьдесят семидесяти"),
    (80, "восемьдесят восьмидесяти"),
    (90, "девяносто девяноста"),
]:
    _reg("t", _v, _f)
for _v, _f in [
    (100, "сто ста"),
    (200, "двести двухсот"),
    (300, "триста трехсот"),
    (400, "четыреста четырехсот"),
    (500, "пятьсот пятисот"),
    (600, "шестьсот шестисот"),
    (700, "семьсот семисот"),
    (800, "восемьсот восьмисот"),
    (900, "девятьсот девятисот"),
]:
    _reg("h", _v, _f)
_reg("s", 1e3, "тысяча тысячи тысяч тысячу тысячей тысячам тысячами")
_reg("s", 1e6, "миллион миллиона миллионов миллионе миллионом миллионам")
_reg("s", 1e9, "миллиард миллиарда миллиардов")
_reg("u", 1.5, "полтора полутора")

DATE_TEXT = re.compile(r"(?<!\d)(\d{1,2})\s+" + MONTH_WORD + r"\s+(\d{4})", re.I)
REF_RE = re.compile(
    r"(?:(?<![а-яё])(?:пп|п|ч|ст|абз|подп|гл|разд)\.\s*|(?<![а-яё])(?:пункт|подпункт|част|стать|абзац|глав|раздел)\w*\s+)"
    r"\d+(?:[.\-]\d+)*(?:\s*(?:,|и|–|-|—)\s*\d+(?:[.\-]\d+)*)*",
    re.I,
)
DOCNUM_RE = re.compile(
    r"(?:№|\bN)\s*[\w\-/.]*\d[\w\-/]*|\b[A-ZА-ЯЁ]{1,4}-\d+(?:[/-]\d+)?\b|\b\d+-(?:ФЗ|П|ПП|ОД)\b|ИНН\s*\d+"
)
DOCNUM_RE_CI = re.compile(DOCNUM_RE.pattern, re.I)
_DOCNUM = DOCNUM_RE_CI if FIX_DOCNUM else DOCNUM_RE
LISTMARK_RE = re.compile(r"(?:(?<=\s)|^)\d{1,2}\)\s")
NUM_RE = re.compile(
    r"(?<![\w.,])(\d{1,3}(?:[  ]\d{3})+|\d+)(?:[.,](\d+))?(?![\d])"
    r"(?:\s*(тыс\w*\.?|млн\.?|миллион\w*|млрд\.?|миллиард\w*))?",
    re.I,
)
NUMWORD_RE = re.compile(r"[а-яё]+", re.I)
UNIT_STEMS = [
    ("рабоч", "wday"),
    ("календар", "cday"),
    ("дн", "day"),
    ("ден", "day"),
    ("сут", "day"),
    ("месяц", "month"),
    ("мес", "month"),
    ("недел", "week"),
    ("год", "year"),
    ("лет", "year"),
    ("час", "hour"),
    ("минут", "minute"),
    ("руб", "rub"),
    ("процент", "pct"),
    ("числ", "dom"),
]
SKIP_TOKENS = {"го", "й", "е", "ти", "х", "ми", "м", "ый", "ой"}


def _mult(word: str | None) -> float:
    if not word:
        return 1.0
    w = norm(word)
    if w.startswith("тыс"):
        return 1e3
    if w.startswith("млрд") or w.startswith("миллиард"):
        return 1e9
    if w.startswith("млн") or w.startswith("миллион"):
        return 1e6
    return 1.0


def _unit_after(t: str, pos: int) -> str | None:
    seg = t[pos : pos + 70]
    if CHECK_FIXES:
        seg = re.split(r"[×*=+÷/]|\s[-−]\s|\d", seg, maxsplit=1)[0]
    for tok in re.findall(r"[а-яa-z]+|%|₽", seg):
        if tok in NUM_WORDS or tok in SKIP_TOKENS:
            continue
        if tok == "%":
            return "pct"
        if tok == "₽":
            return "rub"
        if tok in ("раз", "раза"):
            return "times"
        for stem, unit in UNIT_STEMS:
            if tok.startswith(stem):
                return unit
        return None
    return None


def _strip_dates(text: str) -> tuple[set[tuple[int, int, int]], str]:
    found = set()
    for m in DATE_NUM.finditer(text):
        d, mo, y = (int(x) for x in m.groups())
        found.add((d, mo, y))
    for m in DATE_TEXT.finditer(text):
        mo = month_no(m.group(2))
        if mo:
            found.add((int(m.group(1)), mo, int(m.group(3))))
    text = DATE_NUM.sub(" ", text)
    text = DATE_TEXT.sub(" ", text)
    return found, text


def extract_numbers(text: str, answer: bool = False) -> tuple[list[tuple[float, str, str | None]], set]:
    dates, t = _strip_dates(norm(text))
    if answer:
        t = REF_RE.sub(" ", t)
        t = _DOCNUM.sub(" ", t)
        t = LISTMARK_RE.sub(" ", t)
    else:
        t = _DOCNUM.sub(lambda m: m.group(0) + " " + " ".join(re.findall(r"\d+", m.group(0))), t)
    out = []
    for m in NUM_RE.finditer(t):
        intpart = re.sub(r"[  ]", "", m.group(1))
        v = float(intpart + ("." + m.group(2) if m.group(2) else "")) * _mult(m.group(3))
        out.append((v, m.group(0).strip(), _unit_after(t, m.end())))
    total, cur, seq, last_end = 0.0, 0.0, [], 0
    for m in list(NUMWORD_RE.finditer(t)) + [None]:
        info = NUM_WORDS.get(m.group(0)) if m else None
        if info:
            kind, v = info
            if kind == "s":
                total += (cur or 1.0) * v
                cur = 0.0
            else:
                cur += v
            seq.append(m.group(0))
            last_end = m.end()
            continue
        if seq:
            out.append((total + cur, " ".join(seq), _unit_after(t, last_end)))
            total, cur, seq = 0.0, 0.0, []
    return out, dates


def _key(v: float) -> float:
    return round(v, 6)


def _unit_ok(au: str, cu: str | None) -> bool:
    if au in ("wday", "cday"):
        return cu in (au, "day")
    if au == "day":
        return cu in ("day", "wday", "cday")
    return au == cu


def unverified(answer: str, context: str, question: str) -> list[tuple[str, str]]:
    ctx_items, ctx_dates = extract_numbers(context + "\n" + question)
    units: dict[float, set] = {}
    for v, _, u in ctx_items:
        units.setdefault(_key(v), set()).add(u)
    for d, mo, y in ctx_dates:
        for v in (d, mo, y):
            units.setdefault(_key(v), set()).add(None)
    q_vals = [v for v, _, _ in extract_numbers(question)[0]]
    primary = context.split("=== ЗАКОНЫ", 1)[0] + "\n" + question
    first = sorted({v for v, _, _ in extract_numbers(primary)[0] if 0 < abs(v) < 1e12})
    rest = sorted({v for v, _, _ in ctx_items if 0 < abs(v) < 1e12} - set(first))
    base = (first + rest)[:500]
    ans_items, ans_dates = extract_numbers(answer, answer=True)
    bad = [(f"{d:02d}.{mo:02d}.{y}", "date") for d, mo, y in ans_dates if (d, mo, y) not in ctx_dates]
    for v, span, unit in ans_items:
        keys = [_key(v * k) for k in (1, 1e3, 1e6, 1e-3, 1e-6)]
        if CHECK_FIXES and unit in TERM_UNITS and v == 1 and not re.search(r"\d", span):
            continue
        if v.is_integer() and 1990 <= v <= 2100 and not (CHECK_FIXES and unit is not None):
            if any(k in units for k in keys):
                continue
            bad.append((span, "date"))
            continue
        if unit is None:
            if any(k in units for k in keys) or _derivable(v, base, q_vals):
                continue
        else:
            if any(_unit_ok(unit, cu) for k in keys for cu in units.get(k, ())):
                continue
            if (unit in ("rub", "pct") or v >= 100) and _derivable(v, base, q_vals):
                continue
        bad.append((span, "term" if unit in TERM_UNITS else "amount"))
    return bad


TERM_UNITS = {"day", "wday", "cday", "month", "week", "year", "hour", "minute", "dom", "times"}


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(0.01, abs(b) * 1e-6)


def _derivable(v: float, base: list[float], q: list[float]) -> bool:
    if v == 0:
        return True
    cand = base
    for a in cand:
        for b in cand:
            if (
                _close(v, a * b)
                or _close(v, a * b / 100)
                or (b and _close(v, a / b))
                or _close(v, a * (1 + b / 100))
                or _close(v, a * (1 - b / 100))
            ):
                return True
    for a in q:
        for b in cand + q:
            if _close(v, a + b) or _close(v, a - b) or _close(v, b - a):
                return True
        for b in q:
            for c in cand:
                if _close(v, a * b * c / 100) or _close(v, a * b * c):
                    return True
    return False


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z«\"(])")


def drop_sentences(answer: str, spans: list[str]) -> str:
    sents = SENT_SPLIT.split(answer)
    tail = r"(?![\d]|[.,]\d)" if CHECK_FIXES else r"(?![\d.,])"
    pats = [re.compile(r"(?<![\d.,])" + re.escape(sp) + tail, re.I) for sp in spans if sp]
    keep = [s for s in sents if not any(p.search(norm(s)) for p in pats)]
    if not keep or all(_is_source(s) for s in keep):
        return answer
    out = " ".join(keep).strip()
    return out if len(out.split()) >= 8 else answer


MD_RE = [
    (re.compile(r"\*\*|__|`"), ""),
    (re.compile(r"^#+\s*", re.M), ""),
    (re.compile(r"^\s*[-*•]\s+", re.M), ""),
    (re.compile(r"^\s*Ответ\s*:\s*", re.I), ""),
]


_SOURCE_START = re.compile(r"Источник\s*:", re.I)
_SENT_END = re.compile(r"[.!?](?=\s+[А-ЯЁA-Z«(]|\s*$)")
_ABBR_BEFORE = re.compile(r"(?:^|[\s(«№])(?:г|гг|п|пп|ч|ст|д|ул|обл|ред|им|т|руб|коп|см|др|прим|абз)$", re.I)


def _is_source(sentence: str) -> bool:
    return bool(_SOURCE_START.match(sentence.strip()))


def trim_after_source(text: str) -> str:
    text = text or ""
    starts = list(_SOURCE_START.finditer(text))
    if len(starts) != 1 or not text[: starts[0].start()].strip():
        return text
    for m in _SENT_END.finditer(text, starts[0].end()):
        if m.group(0) == "." and _ABBR_BEFORE.search(text[: m.start()]):
            continue
        end = m.end()
        return text[:end].strip() if text[end:].strip() else text
    return text


def clean_answer(text: str) -> str:
    t = THINK_RE.sub("", text or "")
    for rx, rep in MD_RE:
        t = rx.sub(rep, t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > 2 and t[0] == t[-1] == '"':
        t = t[1:-1].strip()
    if len(t) > ANSWER_MAX_CHARS:
        cut = t[:ANSWER_MAX_CHARS]
        dot = cut.rfind(". ")
        t = cut[: dot + 1] if dot > ANSWER_MAX_CHARS // 2 else cut
    return t


def drop_ref_sentences(answer: str, spans: list[str]) -> str:
    sents = SENT_SPLIT.split(answer)
    keep = [s for s in sents if not any(sp and sp.lower() in s.lower() for sp in spans)]
    if not keep or all(_is_source(s) for s in keep):
        return answer
    out = " ".join(keep).strip()
    return out if len(out.split()) >= 8 else answer
