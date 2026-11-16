from __future__ import annotations

import hashlib
import io
import re
import zipfile
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

from config import CHUNK_FIXES, CHUNK_MAX, DATA_DIR, log
from textproc import CODE_RE, STEMMER, STOPWORDS, WORD_RE, norm, tokenize
from dates import DATE_NUM, DMY_TEXT, fmt_date, mkdate, month_no


@dataclass
class Doc:
    idx: int
    path: str
    kind: str
    label: str
    title: str
    text: str
    status: str = ""
    valid_from: date | None = None
    valid_to: date | None = None
    lost_force: date | None = None
    export_date: str = ""
    art_notes: list = field(default_factory=list)
    removed: list = field(default_factory=list)
    amendments: list = field(default_factory=list)
    body: str = ""
    search_extra: str = ""
    act: str = ""
    article: str = ""


@dataclass
class Chunk:
    idx: int
    doc: int
    label: str
    text: str
    order: int
    search: str = ""
    start: int = -1
    end: int = -1
    clause: str | None = None
    point: str | None = None
    sub: bool = False


def decode_bytes(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def read_docx(raw: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(raw)) as doc:
        xml = doc.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    return "\n".join("".join(x.text or "" for x in p.iter(ns + "t")) for p in root.iter(ns + "p"))


def zip_entry_name(info: zipfile.ZipInfo) -> str:
    name = info.filename
    if not info.flag_bits & 0x800:
        try:
            name = name.encode("cp437").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return name


def candidate_roots() -> list[Path]:
    roots = [DATA_DIR, Path(__file__).resolve().parent / DATA_DIR, Path.cwd() / "data", Path.cwd()]
    seen, out = set(), []
    for r in roots:
        key = str(r.resolve()) if r.exists() else str(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def find_questions() -> Path:
    for root in candidate_roots():
        p = root / "questions.csv"
        if p.is_file():
            return p
    raise FileNotFoundError("Не найден questions.csv")


def find_kb() -> Path | None:
    zips = ("knowledge-base.zip", "knowledge_base.zip", "kb.zip")
    dirs = ("knowledge-base", "knowledge_base", "kb")
    for root in candidate_roots():
        for name in zips:
            if (root / name).is_file():
                return root / name
        for name in dirs:
            if (root / name).is_dir():
                return root / name
    for root in candidate_roots():
        if root.is_dir():
            found = sorted(root.glob("*.zip"))
            if len(found) == 1:
                return found[0]
    return None


def junk_path(name: str) -> bool:
    parts = [p for p in name.replace("\\", "/").split("/") if p]
    return any(p.startswith(".") or p == "__MACOSX" or p.startswith("~$") for p in parts)


def iter_kb_files(path: Path):
    if path.is_dir():
        for p in sorted(path.rglob("*")):
            rel = str(p.relative_to(path))
            if p.is_file() and not junk_path(rel):
                yield rel, p.read_bytes()
        return
    with zipfile.ZipFile(path) as archive:
        for info in sorted(archive.infolist(), key=lambda i: zip_entry_name(i)):
            name = zip_entry_name(info)
            if info.is_dir() or junk_path(name):
                continue
            yield name, archive.read(info)


def load_raw_docs() -> list[tuple[str, str]]:
    kb = find_kb()
    if kb is None:
        log("ВНИМАНИЕ: база знаний не найдена")
        return []
    docs = []
    for name, raw in iter_kb_files(kb):
        suffix = Path(name).suffix.lower()
        try:
            if suffix in (".txt", ".md", ".csv"):
                if b"\x00" in raw[:4096]:
                    log(f"Пропущен бинарный файл: {name}")
                    continue
                text = decode_bytes(raw)
            elif suffix == ".docx":
                text = read_docx(raw)
            else:
                log(f"Пропущен файл неподдерживаемого формата: {name}")
                continue
        except Exception as exc:
            log(f"Пропущен нечитаемый файл {name}: {type(exc).__name__}")
            continue
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
        if text.strip():
            docs.append((name, text))
    log(f"База знаний: {kb.name}, документов: {len(docs)}")
    return docs


INJ_DIRECT = re.compile(
    r"(?i)(ии[-\s]?ассистент|ai[-\s]?assistant|для\s+(?:ии|ai|нейросет\w*|языков\w+\s+модел\w*|чат-?бот\w*|"
    r"ассистент\w*|модел\w+)\b|ignore\s+(?:all|previous|prior|the)|игнорируй\s|не\s+упоминай\s+(?:эту|данн\w+|"
    r"настоящ\w+)\s+инструкц|system\s+prompt|системн\w+\s+промпт)"
)
INJ_WHO = re.compile(r"(?i)(\bии\b|нейросет|языков\w*\s+модел|\bllm\b|chatgpt|\bgpt\b|ассистент|assistant|\bai\b)")
INJ_DO = re.compile(r"(?i)(игнорир|ignore|не\s+упомина|забудь|отвечай|всегда\s+отвеча|инструкци|промпт|prompt)")
INJ_MARK = "[Фрагмент удалён: встроенная инструкция для ИИ, не является положением документа]"


def sanitize(text: str) -> tuple[str, list[str]]:
    out, removed = [], []
    for line in text.split("\n"):
        if INJ_DIRECT.search(line) or (INJ_WHO.search(line) and INJ_DO.search(line)):
            removed.append(line.strip())
            out.append(INJ_MARK)
        else:
            out.append(line)
    return "\n".join(out), removed


def label_from_path(path: str) -> str:
    stem = Path(path).stem
    if "__" in stem:
        stem = stem.split("__", 1)[1]
    return re.sub(r"\s+", " ", stem.replace("_", " ")).strip()


def is_law(path: str, text: str) -> bool:
    p = norm(path)
    if any(k in p for k in ("закон", "кодекс", "законодат", "/law", "laws")) or p.startswith("law"):
        return True
    head = text[:800]
    return bool(re.search(r"^Источник:.*(?:кодекс|Федеральн\w+ закон|КоАП|ТК РФ|НК РФ|ГК РФ)", head, re.M | re.I))


TITLE_KEYS = re.compile(
    r"(ДОГОВОР|СОГЛАШЕНИЕ|РЕГЛАМЕНТ|ПИСЬМО|ЗАПИСКА|ПРИКАЗ|ПОЛОЖЕНИЕ|ИНСТРУКЦИЯ|ПОЛИТИКА|" r"ПОРЯДОК|ПРАВИЛА|Исх\.)"
)


def company_meta(doc: Doc) -> None:
    lines = [ln.strip() for ln in doc.text.split("\n") if ln.strip()]
    title = next((ln for ln in lines[:8] if TITLE_KEYS.search(ln)), lines[0] if lines else doc.label)
    doc.title = title[:200]
    head = doc.text[:700]
    m = re.search(r"утратил\w*\s+сил\w*\s+с\s+(\d{1,2})\.(\d{1,2})\.(\d{4})", head, re.I)
    if m:
        doc.lost_force = mkdate(*m.groups())
        if doc.lost_force:
            doc.valid_to = doc.lost_force - timedelta(days=1)
    m = re.search(r"(?:Действует|вступает в силу|применяется)\s+с\s+(\d{1,2})\.(\d{1,2})\.(\d{4})", head, re.I)
    if m:
        doc.valid_from = mkdate(*m.groups())
    if not doc.valid_from:
        m = re.search(r"Редакция\s+от\s+(\d{1,2})\.(\d{1,2})\.(\d{4})", head, re.I)
        if m:
            doc.valid_from = mkdate(*m.groups())
    parts = []
    if doc.lost_force:
        parts.append(f"УТРАТИЛ СИЛУ с {fmt_date(doc.lost_force)}")
        if doc.valid_from:
            parts.append(f"действовал с {fmt_date(doc.valid_from)} по {fmt_date(doc.valid_to)}")
    elif "утратил" in norm(doc.label):
        parts.append("утратил силу")
    elif doc.valid_from and re.search(r"редакц|регламент|положени|политик|порядок|правил", norm(doc.label + doc.title)):
        parts.append(f"действующая редакция, действует с {fmt_date(doc.valid_from)}")
    doc.status = "; ".join(parts)


def company_chunks(doc: Doc, start_idx: int) -> list[Chunk]:
    chunks, buf, order = [], [], 0
    section = ""

    def flush():
        nonlocal buf, order
        if buf:
            body = "\n".join(buf)
            nums = re.findall(r"^(\d+(?:\.\d+)+)\.", body, re.M)
            clause = f"п. {nums[0]}" + (f"–{nums[-1]}" if len(nums) > 1 else "") if nums else None
            lab = doc.label + (f", {clause}" if clause else "")
            text = (section + "\n" if section and not body.startswith(section) else "") + body
            chunks.append(Chunk(start_idx + len(chunks), doc.idx, lab, text, order, clause=clause))
            order += 1
            buf = []

    for line in doc.text.split("\n"):
        s = line.strip()
        if not s:
            continue
        if re.match(r"^\d+\.\s+[А-ЯЁA-Z]", s) and len(s) < 90:
            flush()
            section = s
            buf.append(s)
            continue
        if sum(len(x) for x in buf) + len(s) > 900:
            flush()
        buf.append(s)
    flush()
    return chunks


NOTE_RE = re.compile(
    r"^(?:(?:Пункт|Подпункт|Часть|Части|Статья|Абзац|Примечани\w*|Наименование|Раздел|Глава)\s+[\w.,()\s-]{0,40}?"
    r"(?:изменен|изложен|дополнен|утратил|признан|приостановлен|вступа|не применя)|Положения\s+(?:пункта|подпункта|"
    r"части|статьи|абзаца)|Федеральн\w+\s+закон\w*\s+от\s+\d|О\s+(?:конституционно|проверке|применении|порядке)|"
    r"См\.\s|Информация об изменениях|[CС]рок давности привлечения)",
    re.I,
)
POINT_RE = re.compile(r"^(\d+(?:\.\d+)*(?:-\d+)?)\.\s+\S")
SUB_RE = re.compile(r"^(\d+(?:\.\d+)?(?:-\d+)?)\)\s*\S")
HEADING_RE = re.compile(r"^Статья\s+(\d+(?:\.\d+)*)\.\s*(.*)$")


def law_act(source: str, path: str) -> str:
    s = source
    if re.search(r"Трудов\w+\s+кодекс|ТК РФ", s):
        return "ТК РФ"
    if re.search(r"Налогов\w+\s+кодекс|НК РФ", s):
        return "НК РФ"
    if re.search(r"административных правонарушениях|КоАП", s):
        return "КоАП РФ"
    if re.search(r"Гражданск\w+\s+кодекс|ГК РФ", s):
        return "ГК РФ"
    m = re.search(r"Федеральн\w+\s+закон\w*\s+от\s+([\d.]+)\s+(?:N|№)\s*(\d+-ФЗ)\s*[\"«]([^\"»]+)[\"»]", s)
    if m:
        return f"Федеральный закон от {m.group(1)} № {m.group(2)} «{m.group(3)}»"
    m = re.search(r"(\d+-ФЗ)", s + " " + path)
    if m:
        return f"Федеральный закон № {m.group(1)}"
    return label_from_path(path)


def _split_sentences(par: str, limit: int) -> list[str]:
    parts, cur = [], ""
    for sent in re.split(r"(?<=[.;:])\s+", par):
        if cur and len(cur) + len(sent) + 1 > limit:
            parts.append(cur)
            cur = sent
        else:
            cur = (cur + " " + sent).strip()
    if cur:
        parts.append(cur)
    out = []
    for p in parts:
        while len(p) > limit * 1.5:
            out.append(p[:limit])
            p = p[limit:]
        out.append(p)
    return out


NK_CHAPTERS = [
    (106, 115, "налоговая ответственность, налоговые правонарушения, налоговые санкции, штрафы"),
    (116, 142, "налоговые правонарушения и ответственность за их совершение"),
    (143, 178, "налог на добавленную стоимость (НДС), глава 21 НК РФ"),
    (179, 206, "акцизы, глава 22 НК РФ"),
    (207, 233, "налог на доходы физических лиц (НДФЛ), глава 23 НК РФ"),
    (246, 333, "налог на прибыль организаций, глава 25 НК РФ"),
    (346, 346, "специальные налоговые режимы"),
]


def law_topic(act: str, art_no: str) -> str:
    if act != "НК РФ" or not art_no:
        return ""
    try:
        n = int(art_no.split(".")[0])
    except ValueError:
        return ""
    for lo, hi, topic in NK_CHAPTERS:
        if lo <= n <= hi:
            return topic
    return ""


def split_long(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    cur = ""
    pars: list[str] = []
    note_buf = ""
    for par in text.split("\n"):
        par = par.strip()
        if not par:
            continue
        if par.startswith("[Пометка редакции:"):
            note_buf = (note_buf + "\n" + par) if note_buf else par
            continue
        pars.append((note_buf + "\n" + par) if note_buf else par)
        note_buf = ""
    if note_buf:
        pars.append(note_buf)
    for par in pars:
        if len(par) > limit:
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.extend(_split_sentences(par, limit))
            continue
        if cur and len(cur) + len(par) + 1 > limit:
            pieces.append(cur)
            cur = par
        else:
            cur = (cur + "\n" + par) if cur else par
    if cur:
        pieces.append(cur)
    return pieces


def parse_law(doc: Doc, start_idx: int) -> list[Chunk]:
    act, art_no, body = _law_header(doc)
    base = doc.label
    word = "п." if act.startswith("НК") else "ч."
    units = _merge_heading(_law_units(doc, body))
    return _law_chunks(doc, base, word, units, start_idx)


def _law_header(doc: Doc) -> tuple[str, str, list[str]]:
    lines = doc.text.split("\n")
    source = ""
    body = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("Источник:"):
            source = s[len("Источник:") :].strip()
        elif s.startswith("URL:"):
            continue
        elif s.startswith("Дата выгрузки:"):
            doc.export_date = s.split(":", 1)[1].strip()
        else:
            body.append(s)
    act = law_act(source or doc.label, doc.path)
    art_no, art_title = "", ""
    m = re.search(r"Стать[яи]\s+(\d+(?:\.\d+)*)", source)
    if m:
        art_no = m.group(1)
    for s in body:
        hm = HEADING_RE.match(s)
        if hm:
            art_no = art_no or hm.group(1)
            if len(hm.group(2)) <= 150 and not re.search(r"\d{3}", hm.group(2)):
                art_title = hm.group(2).strip().rstrip(".")
            break
    base = f"{act}, ст. {art_no}" if art_no else act
    doc.label = base
    doc.act, doc.article = act, art_no
    doc.title = f"{base} «{art_title}»" if art_title else base
    topic = law_topic(act, art_no)
    if topic:
        doc.search_extra = f"{doc.title} — {topic}"
        doc.title += f" — {topic.split(', глава')[0]}"
    return act, art_no, body


def _law_units(doc: Doc, body: list[str]) -> list[dict]:
    numbered = sum(1 for s in body if POINT_RE.match(s)) >= 2

    units: list[dict] = []
    pending: list[str] = []
    point = None
    heading_seen = False
    cur = None
    body_out: list[str] = []
    for s in body:
        if not s:
            continue
        if NOTE_RE.match(s):
            if not heading_seen:
                doc.art_notes.append(s)
            else:
                pending.append(s)
                body_out.append(f"[Пометка редакции: {s}]")
            continue
        body_out.append(s)
        if HEADING_RE.match(s):
            heading_seen = True
        pm = POINT_RE.match(s) if numbered else None
        sm = SUB_RE.match(s)
        if pm:
            point = pm.group(1)
            cur = {"point": point, "sub": None, "notes": pending, "lines": [s]}
            units.append(cur)
        elif sm and point is not None:
            cur = {"point": point, "sub": sm.group(1), "notes": pending, "lines": [s]}
            units.append(cur)
        elif cur is None or pending:
            cur = {"point": point, "sub": None, "notes": pending, "lines": [s]}
            units.append(cur)
        else:
            cur["lines"].append(s)
        pending = []
    if pending and units:
        units[-1]["lines"].extend(f"[Пометка редакции: {p}]" for p in pending)
    doc.body = "\n".join(body_out)
    return units


def _merge_heading(units: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for u in units:
        if (
            merged
            and len(merged[-1]["lines"]) == 1
            and not merged[-1]["notes"]
            and HEADING_RE.match(merged[-1]["lines"][0])
            and (merged[-1]["point"] == u["point"] or (CHUNK_FIXES and merged[-1]["point"] is None))
        ):
            u = {**u, "lines": merged[-1]["lines"] + u["lines"]}
            merged[-1] = u
        else:
            merged.append(u)
    return merged


def _unit_text(u: dict) -> str:
    notes = "".join(f"[Пометка редакции: {n}]\n" for n in u["notes"])
    return notes + "\n".join(u["lines"])


def _law_chunks(doc: Doc, base: str, word: str, units: list[dict], start_idx: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    order = 0

    def add(label: str, text: str, point: str | None, sub: bool) -> None:
        nonlocal order
        clause = label[len(base) + 2 :] if point else None
        chunks.append(
            Chunk(start_idx + len(chunks), doc.idx, label, text.strip(), order, clause=clause, point=point, sub=sub)
        )
        order += 1

    groups: list[tuple[str | None, list[dict]]] = []
    for u in units:
        if groups and groups[-1][0] == u["point"]:
            groups[-1][1].append(u)
        else:
            groups.append((u["point"], [u]))

    for point, us in groups:
        plabel = f"{base}, {word} {point}" if point else base
        full = "\n".join(_unit_text(u) for u in us)
        if len(full) <= CHUNK_MAX:
            add(plabel, full, point, False)
            continue
        intro = us[0]["lines"][0]
        if CHUNK_FIXES:
            intro = next((ln for ln in us[0]["lines"] if not HEADING_RE.match(ln)), intro)
        intro_short = intro if len(intro) <= 220 else intro[:220] + "…"
        buf: list[dict] = []
        size = 0

        def flush_buf(first: bool) -> None:
            if not buf:
                return
            subs = [u["sub"] for u in buf if u["sub"]]
            lab = plabel + (f", пп. {subs[0]}" + (f"–{subs[-1]}" if len(subs) > 1 else "") if subs and point else "")
            text = "\n".join(_unit_text(u) for u in buf)
            if not first and point:
                text = f"(продолжение {word} {point}: «{intro_short}»)\n" + text
            for i, piece in enumerate(split_long(text, CHUNK_MAX)):
                if i and point:
                    if CHUNK_FIXES:
                        piece = f"(продолжение {word} {point}{lab[len(plabel):]}: «{intro_short}»)\n" + piece
                    else:
                        piece = f"(продолжение {word} {point})\n" + piece
                add(lab, piece, point, bool(subs and point))

        first = True
        for u in us:
            t = _unit_text(u)
            if buf and size + len(t) > CHUNK_MAX:
                flush_buf(first)
                first = False
                buf, size = [], 0
            buf.append(u)
            size += len(t)
        flush_buf(first)
    return chunks


@dataclass
class KB:
    docs: list[Doc]
    chunks: list[Chunk]

    @property
    def company(self) -> list[Doc]:
        return [d for d in self.docs if d.kind == "company"]

    @property
    def laws(self) -> list[Doc]:
        return [d for d in self.docs if d.kind == "law"]


def build_kb() -> KB:
    docs: list[Doc] = []
    chunks: list[Chunk] = []
    for path, raw_text in load_raw_docs():
        text, removed = sanitize(raw_text)
        kind = "law" if is_law(path, text) else "company"
        doc = Doc(
            idx=len(docs),
            path=path,
            kind=kind,
            label=label_from_path(path),
            title="",
            text=text.strip(),
            removed=removed,
        )
        if removed:
            log(f"Удалены встроенные инструкции ({len(removed)}) в документе: {doc.label}")
        if kind == "company":
            company_meta(doc)
            doc.amendments = company_amendments(doc)
            new = company_chunks(doc, len(chunks))
        else:
            new = parse_law(doc, len(chunks))
        docs.append(doc)
        for c in new:
            c.search = f"{doc.search_extra or doc.title}. {c.label}. {c.text}"
        chunks.extend(new)
    log(
        f"Документов компании: {sum(d.kind == 'company' for d in docs)}, законов: "
        f"{sum(d.kind == 'law' for d in docs)}, фрагментов: {len(chunks)}"
    )
    for d in docs:
        locate_chunks(d, [c for c in chunks if c.doc == d.idx])
    return KB(docs, chunks)


def _title_stems(d: Doc) -> set[str]:
    words = [w for w in WORD_RE.findall(norm(d.title)) if not w.isdigit() and len(w) > 2 and w not in STOPWORDS]
    return {STEMMER.stem(w) for w in words}


def related_docs(kb: KB, d: Doc) -> list[Doc]:
    codes_d = set(CODE_RE.findall(norm(d.text[:400])))
    stems_d = _title_stems(d)
    out = []
    for o in kb.company:
        if o is d:
            continue
        codes_o = set(CODE_RE.findall(norm(o.text[:400])))
        if (codes_d & set(CODE_RE.findall(norm(o.text)))) or (codes_o & set(CODE_RE.findall(norm(d.text)))):
            out.append(o)
            continue
        stems_o = _title_stems(o)
        if len(stems_d) >= 3 and (stems_d == stems_o or stems_d <= set(tokenize(o.text))):
            out.append(o)
        elif len(stems_o) >= 3 and stems_o <= set(tokenize(d.text)):
            out.append(o)
    return out


_TRANSLIT = dict(
    zip(
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
        [
            "a",
            "b",
            "v",
            "g",
            "d",
            "e",
            "e",
            "zh",
            "z",
            "i",
            "y",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "r",
            "s",
            "t",
            "u",
            "f",
            "kh",
            "ts",
            "ch",
            "sh",
            "shch",
            "",
            "y",
            "",
            "e",
            "yu",
            "ya",
        ],
    )
)


def slug(text: str) -> str:
    s = "".join(_TRANSLIT.get(ch, ch) for ch in text.lower())
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-") or "doc"


def doc_ids(kb: KB) -> dict[int, str]:
    out: dict[int, str] = {}
    used: set[str] = set()
    for d in sorted(kb.docs, key=lambda d: d.path):
        base = slug(label_from_path(d.path))
        sid, k = base, 2
        while sid in used:
            sid, k = f"{base}-{k}", k + 1
        used.add(sid)
        out[d.idx] = sid
    return out


def doc_type(d: Doc) -> str:
    if d.kind == "law":
        return "статья закона"
    t = (d.title + " " + d.label).upper()
    for key, name in (
        ("ДОПОЛНИТЕЛЬНОЕ СОГЛАШЕНИЕ", "доп. соглашение"),
        ("ДОП СОГЛАШЕНИЕ", "доп. соглашение"),
        ("РЕГЛАМЕНТ", "регламент"),
        ("ЗАПИСКА", "служебная записка"),
        ("ПИСЬМО", "письмо"),
        ("ИСХ.", "письмо"),
        ("ДОГОВОР", "договор"),
    ):
        if key in t:
            return name
    return "документ"


_COMPANY_CLAUSE = re.compile(r"^(\d+(?:\.\d+)*)\.\s+\S")


def _line_spans(text: str) -> list[tuple[int, int, str]]:
    spans, pos = [], 0
    for line in text.split("\n"):
        spans.append((pos, pos + len(line), line))
        pos += len(line) + 1
    return spans


def doc_clauses(d: Doc) -> list[dict]:
    lines = _line_spans(d.text)
    items: list[dict] = []
    if d.kind == "company":
        for s, e, ln in lines:
            m = _COMPANY_CLAUSE.match(ln.strip()) if ln.strip() == ln.lstrip() else None
            if m:
                items.append(
                    {
                        "clause": m.group(1),
                        "level": m.group(1).count(".") + 1,
                        "start": s,
                        "line_end": e,
                        "title": ln.strip()[:120],
                    }
                )
    else:
        numbered = sum(1 for _, _, ln in lines if POINT_RE.match(ln.strip())) >= 2
        word = "п." if d.label.startswith("НК") else "ч."
        point = None
        for s, e, ln in lines:
            st = ln.strip()
            pm = POINT_RE.match(st) if numbered else None
            sm = SUB_RE.match(st)
            if pm and not NOTE_RE.match(st):
                point = pm.group(1)
                items.append({"clause": f"{word} {point}", "level": 1, "start": s, "line_end": e, "title": st[:120]})
            elif sm and point is not None and d.label.startswith("НК"):
                items.append(
                    {
                        "clause": f"{word} {point}, пп. {sm.group(1)}",
                        "level": 2,
                        "start": s,
                        "line_end": e,
                        "title": st[:120],
                    }
                )
    for i, it in enumerate(items):
        end = len(d.text)
        for nxt in items[i + 1 :]:
            if nxt["level"] <= it["level"]:
                end = nxt["start"]
                break
        it["end"] = len(d.text[:end].rstrip())
        del it["line_end"]
    return items


def doc_markup(d: Doc) -> dict:
    notes, removed = [], []
    rem = iter(d.removed)
    for s, e, ln in _line_spans(d.text):
        st = ln.strip()
        if d.kind == "law" and st and NOTE_RE.match(st):
            notes.append({"start": s, "end": e, "text": ln})
        if st == INJ_MARK:
            removed.append(
                {
                    "start": s,
                    "end": e,
                    "original": next(rem, ""),
                    "note": "Встроенная инструкция для ИИ удалена: не является положением документа",
                }
            )
    return {"edition_notes": notes, "removed_instructions": removed}


def kb_hash(kb: KB) -> str:
    h = hashlib.sha256()
    for d in sorted(kb.docs, key=lambda d: d.path):
        h.update(d.path.encode("utf-8") + b"\0" + hashlib.sha256(d.text.encode("utf-8")).digest())
    return h.hexdigest()


_NOTE_LINE = re.compile(r"^\[Пометка редакции: (.*)\]$")


def _source_lines(chunk_text: str) -> list[str]:
    out = []
    for ln in chunk_text.split("\n"):
        ln = ln.strip()
        if not ln or ln.startswith("(продолжение "):
            continue
        m = _NOTE_LINE.match(ln)
        out.append(m.group(1) if m else ln)
    return out


def locate_chunks(d: Doc, chunks: list[Chunk]) -> None:
    text = d.text
    cursor = 0
    for c in sorted(chunks, key=lambda c: c.order):
        lines = _source_lines(c.text)
        if not lines:
            m = re.search(r"«(.+?)…?»\)", c.text)
            lines = [m.group(1).rstrip("…").strip()] if m else []
        found = []
        for ln in lines:
            p = text.find(ln, cursor)
            if p >= 0:
                found.append((p, p + len(ln)))
        if not found:
            for ln in lines:
                p = text.find(ln)
                if p >= 0:
                    found.append((p, p + len(ln)))
        if not found:
            continue
        c.start = min(a for a, _ in found)
        c.end = max(b for _, b in found)
        cursor = c.start


_DATE_ANY = r"(\d{1,2}\.\d{1,2}\.\d{4}|\d{1,2}\s+[а-яё]+\s+\d{4})"


def _parse_date(raw: str):
    m = DATE_NUM.search(raw)
    if m:
        return mkdate(*m.groups())
    m = DMY_TEXT.search(raw)
    if m and month_no(m.group(2)):
        return mkdate(m.group(1), str(month_no(m.group(2))), m.group(3))
    return None


def _sentences(text: str) -> list[str]:
    out = []
    for line in text.split("\n"):
        out += [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[А-ЯЁ«])", line) if x.strip()]
    return out


def company_amendments(d: Doc) -> list[dict]:
    target = ""
    m = re.search(r"в дополнение к [^.]*?(п\.\s*[\d.]+[^.()]*?(?:\([^)]*\))?)", d.text)
    if m:
        target = "дополняет " + m.group(1).strip()
    m = re.search(r"[Ии]зложить пункт\s+([\d.]+)\s+договора", d.text)
    if m:
        parent = re.search(r"к договору[^\n]*?№\s*([\w\-/]+)", d.text)
        target = f"меняет п. {m.group(1)} договора" + (f" № {parent.group(1)}" if parent else "")
    out = []
    for sent in _sentences(d.text):
        m = re.match(rf"[Сс]\s+{_DATE_ANY}\s*(?:г\.)?", sent)
        if m and _parse_date(m.group(1)):
            out.append({"kind": "from", "date": _parse_date(m.group(1)), "condition": sent[:220], "target": target})
            continue
        m = re.search(rf"(?:подписан\w*|заключ\w*)\s+после\s+{_DATE_ANY}", sent)
        if m and _parse_date(m.group(1)):
            out.append({"kind": "after", "date": _parse_date(m.group(1)), "condition": sent[:220], "target": target})
    return out


_XR_ANCHOR = re.compile(r"настоящ(?:ей|его)\s+(статьи|пункта|подпункта|Кодекса|Федерального\s+закона|Закона)")
_XR_NUM = r"\d+(?:\.\d+)*(?:-\d+)?"
_XR_TOKEN = re.compile(
    rf"(подпункт|пункт|част|стать|абзац)\w*\s+({_XR_NUM}(?:\.?\s*(?:,|\bили\b|\bи\b|\bлибо\b|\s[-–—]\s|[–—])\s*{_XR_NUM})*)\.?",
    re.I,
)
_XR_GAP = re.compile(r"^[\s,]*(?:(?:и|или|либо)\s+)?$")
_XR_LEVEL = {"абзац": 0, "подпункт": 1, "пункт": 2, "част": 3, "стать": 4}


def _xr_nums(raw: str) -> list[str]:
    out: list[str] = []
    for part in re.split(r"\s*(?:,|\bили\b|\bи\b|\bлибо\b)\s*", raw.strip().rstrip(".")):
        m = re.fullmatch(rf"({_XR_NUM})\s*(?:\s[-–—]\s|[–—])\s*({_XR_NUM})", part.strip())
        if m and m.group(1).isdigit() and m.group(2).isdigit() and 0 < int(m.group(2)) - int(m.group(1)) <= 20:
            out += [str(i) for i in range(int(m.group(1)), int(m.group(2)) + 1)]
        elif m:
            out += [m.group(1), m.group(2)]
        elif re.fullmatch(_XR_NUM, part.strip()):
            out.append(part.strip())
    return out


def _xr_groups(segment: str) -> list[dict]:
    toks = list(_XR_TOKEN.finditer(segment))
    chain: list[re.Match] = []
    end = len(segment)
    for t in reversed(toks):
        if not _XR_GAP.match(segment[t.end() : end]):
            break
        chain.insert(0, t)
        end = t.start()
    groups: list[dict] = []
    cur: dict = {}
    for t in chain:
        kind = t.group(1).lower()
        lvl = _XR_LEVEL[kind]
        if cur and lvl <= max(_XR_LEVEL[k] for k in cur):
            groups.append(cur)
            cur = {}
        cur[kind] = _xr_nums(t.group(2))
    if cur:
        groups.append(cur)
    return groups


def cross_refs(kb: KB) -> dict[int, list[dict]]:
    cached = getattr(kb, "_cross_refs", None)
    if cached is not None:
        return cached
    act_of = {d.idx: d.act for d in kb.laws}
    art_of = {d.idx: d.article for d in kb.laws}
    by_art = {(act_of[i], art_of[i]): i for i in act_of if art_of[i]}
    by_point: dict[tuple[int, str | None], list[Chunk]] = {}
    for c in kb.chunks:
        if c.doc in act_of:
            by_point.setdefault((c.doc, c.point), []).append(c)

    def targets(di: int, point: str | None, sub: str | None) -> list[int]:
        chs = by_point.get((di, point), [])
        if not chs:
            return []
        if sub:
            for c in chs:
                if re.search(rf"(?m)^{re.escape(sub)}\)", c.text):
                    return [c.idx]
            return []
        return [c.idx for c in chs] if sum(len(c.text) for c in chs) <= 3000 else []

    out: dict[int, list[dict]] = {}
    for c in kb.chunks:
        if c.doc not in act_of:
            continue
        act, own_art, own_point = act_of[c.doc], art_of[c.doc], c.point
        text = re.sub(r"\[Пометка редакции:[^\]]*\]", lambda m: " " * len(m.group(0)), c.text)
        prev = 0
        for a in _XR_ANCHOR.finditer(text):
            seg_start = max(prev, a.start() - 220)
            prev = a.end()
            segment = text[seg_start : a.start()]
            cut = max(segment.rfind(";"), segment.rfind("\n"))
            segment = segment[cut + 1 :]
            what = a.group(1).lower()
            refs_t: list[int] = []
            absent: list[tuple[str, str]] = []
            for g in _xr_groups(segment):
                arts = g.get("стать")
                if what == "статьи":
                    arts = [own_art]
                elif what == "пункта":
                    if "подпункт" not in g or "пункт" in g:
                        continue
                    g = {**g, "пункт": [own_point] if own_point else []}
                    arts = [own_art]
                elif what == "подпункта":
                    continue
                if not arts:
                    continue
                if "част" in g:
                    points, subs = g["част"], g.get("пункт") or g.get("подпункт") or [None]
                else:
                    points, subs = g.get("пункт") or [None], g.get("подпункт") or [None]
                for art in arts:
                    di = by_art.get((act, art))
                    if di is None:
                        if (act, art) not in absent:
                            absent.append((act, art))
                        continue
                    if points == [None]:
                        continue
                    for p in points:
                        for s in subs:
                            for t in targets(di, p, s):
                                if t != c.idx and t not in refs_t:
                                    refs_t.append(t)
            if refs_t or absent:
                out.setdefault(c.idx, []).append(
                    {"text": (segment.strip() + " " + a.group(0))[-200:], "targets": refs_t, "absent": absent}
                )
    kb._cross_refs = out
    return out
