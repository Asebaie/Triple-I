from __future__ import annotations

import re
import threading

from config import CHECK_FIXES


class RussianStemmer:
    V = set("аеиоуыэюя")
    PERF = sorted(
        [
            ("в", 1),
            ("вши", 1),
            ("вшись", 1),
            ("ив", 2),
            ("ивши", 2),
            ("ившись", 2),
            ("ыв", 2),
            ("ывши", 2),
            ("ывшись", 2),
        ],
        key=lambda x: -len(x[0]),
    )
    REFL = ("ся", "сь")
    ADJ = sorted(
        [
            "ее",
            "ие",
            "ые",
            "ое",
            "ими",
            "ыми",
            "ей",
            "ий",
            "ый",
            "ой",
            "ем",
            "им",
            "ым",
            "ом",
            "его",
            "ого",
            "ему",
            "ому",
            "их",
            "ых",
            "ую",
            "юю",
            "ая",
            "яя",
            "ою",
            "ею",
        ],
        key=len,
        reverse=True,
    )
    PART = sorted(
        [("ем", 1), ("нн", 1), ("вш", 1), ("ющ", 1), ("щ", 1), ("ивш", 2), ("ывш", 2), ("ующ", 2)],
        key=lambda x: -len(x[0]),
    )
    VERB = sorted(
        [
            (s, 1)
            for s in [
                "ла",
                "на",
                "ете",
                "йте",
                "ли",
                "й",
                "л",
                "ем",
                "н",
                "ло",
                "но",
                "ет",
                "ют",
                "ны",
                "ть",
                "ешь",
                "нно",
            ]
        ]
        + [
            (s, 2)
            for s in [
                "ила",
                "ыла",
                "ена",
                "ейте",
                "уйте",
                "ите",
                "или",
                "ыли",
                "ей",
                "уй",
                "ил",
                "ыл",
                "им",
                "ым",
                "ен",
                "ило",
                "ыло",
                "ено",
                "ят",
                "ует",
                "уют",
                "ит",
                "ыт",
                "ены",
                "ить",
                "ыть",
                "ишь",
                "ую",
                "ю",
            ]
        ],
        key=lambda x: -len(x[0]),
    )
    NOUN = sorted(
        [
            "а",
            "ев",
            "ов",
            "ие",
            "ье",
            "е",
            "иями",
            "ями",
            "ами",
            "еи",
            "ии",
            "и",
            "ией",
            "ей",
            "ой",
            "ий",
            "й",
            "иям",
            "ям",
            "ием",
            "ем",
            "ам",
            "ом",
            "о",
            "у",
            "ах",
            "иях",
            "ях",
            "ы",
            "ь",
            "ию",
            "ью",
            "ю",
            "ия",
            "ья",
            "я",
        ],
        key=len,
        reverse=True,
    )

    def __init__(self) -> None:
        self.cache: dict[str, str] = {}
        self.lock = threading.Lock()

    def _regions(self, w: str) -> tuple[int, int]:
        n = len(w)

        def gopast(start: int, vowel: bool) -> int:
            for i in range(start, n):
                if (w[i] in self.V) == vowel:
                    return i + 1
            return -1

        pv = gopast(0, True)
        if pv < 0:
            return n, n
        c = gopast(pv, False)
        if c < 0:
            return pv, n
        c = gopast(c, True)
        if c < 0:
            return pv, n
        c = gopast(c, False)
        return pv, (n if c < 0 else c)

    @staticmethod
    def _among(w: str, rv: int, items):
        for suf, grp in items:
            if w.endswith(suf) and len(w) - len(suf) >= rv:
                if grp == 1:
                    pos = len(w) - len(suf) - 1
                    if pos >= rv and w[pos] in "ая":
                        return suf
                    return None
                return suf
        return None

    def stem(self, word: str) -> str:
        w = word.lower().replace("ё", "е")
        hit = self.cache.get(w)
        if hit is not None:
            return hit
        res = self._stem(w)
        if len(self.cache) < 200000:
            self.cache[w] = res
        return res

    def _stem(self, w: str) -> str:
        rv, r2 = self._regions(w)
        if rv >= len(w):
            return w
        suf = self._among(w, rv, self.PERF)
        if suf:
            w = w[: -len(suf)]
        else:
            for s in self.REFL:
                if w.endswith(s) and len(w) - len(s) >= rv:
                    w = w[: -len(s)]
                    break
            removed = False
            for s in self.ADJ:
                if w.endswith(s) and len(w) - len(s) >= rv:
                    w = w[: -len(s)]
                    p = self._among(w, rv, self.PART)
                    if p:
                        w = w[: -len(p)]
                    removed = True
                    break
            if not removed:
                v = self._among(w, rv, self.VERB)
                if v:
                    w = w[: -len(v)]
                    removed = True
            if not removed:
                for s in self.NOUN:
                    if w.endswith(s) and len(w) - len(s) >= rv:
                        w = w[: -len(s)]
                        break
        if w.endswith("и") and len(w) - 1 >= rv:
            w = w[:-1]
        for s in ("ость", "ост"):
            if w.endswith(s) and len(w) - len(s) >= max(rv, r2):
                w = w[: -len(s)]
                break
        if w.endswith("ейше") and len(w) - 4 >= rv:
            w = w[:-4]
            if w.endswith("нн") and len(w) - 2 >= rv:
                w = w[:-1]
        elif w.endswith("ейш") and len(w) - 3 >= rv:
            w = w[:-3]
            if w.endswith("нн") and len(w) - 2 >= rv:
                w = w[:-1]
        elif w.endswith("нн") and len(w) - 2 >= rv:
            w = w[:-1]
        elif w.endswith("ь") and len(w) - 1 >= rv:
            w = w[:-1]
        return w


STEMMER = RussianStemmer()

STOPWORDS = set("""
а без более бы был была были было быть в вам вас весь во вот все всего всех вы где да даже для до его ее если
есть еще же за здесь и из или им их к как ко когда кто ли либо мне может мы на над надо наш не него нее нет ни
них но ну о об однако он она они оно от очень по под при с со так также такой там те тем то того тоже той только
том ты у уже хотя чего чей чем что чтобы чье чья эта эти это я ли нам нас ваш ваша наши вами какой какая какие
каков сколько можно нужно ли этот этой этого этом этих эту""".split())

WORD_RE = re.compile(r"[a-zа-я0-9]+")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
CODE_RE = re.compile(r"\b[a-zа-я]{1,4}-\d+(?:[/-]\d+)?\b")


def norm(text: str) -> str:
    return text.lower().replace("ё", "е")


def tokenize(text: str) -> list[str]:
    t = norm(text)
    out = []
    for m in CODE_RE.finditer(t):
        out.append(m.group(0).replace("/", "-"))
    for w in WORD_RE.findall(t):
        if w.isdigit():
            out.append(w)
        elif len(w) >= 2 and w not in STOPWORDS:
            out.append(STEMMER.stem(w))
    return out


SYNONYMS = {
    "главбух": "главный бухгалтер",
    "испытательн": "испытание испытания",
    "декрет": "беременность роды",
    "сверхурочк": "сверхурочная работа",
    "переработк": "сверхурочная работа",
    "ндфл": "налог на доходы физических лиц налоговая ставка",
    "ндс": "налог на добавленную стоимость налоговая ставка",
    "налог на прибыль": "налог на прибыль организаций налоговая ставка",
    "персданн": "персональные данные",
    "персональн": "персональные данные субъект оператор",
    "мрот": "минимальный размер оплаты труда",
    "зарплат": "заработная плата оплата труда",
    "штраф": "штраф ответственность санкция",
    "сезонн": "сезонные работы срочный трудовой договор",
    "аванс": "предоплата",
    "предоплат": "аванс",
    "упд": "универсальный передаточный документ",
    "временн": "срочный трудовой договор",
    "упрощенк": "упрощенная система налогообложения",
    "освобожден": "освобождение от исполнения обязанностей налогоплательщика",
    "смягча": "смягчающие обстоятельства",
}


def expand_query(q: str) -> str:
    nq = norm(q)
    if CHECK_FIXES:
        extra = [v for k, v in SYNONYMS.items() if re.search(r"(?<![а-яёa-z])" + re.escape(k), nq)]
    else:
        extra = [v for k, v in SYNONYMS.items() if k in nq]
    return q + (" " + " ".join(extra) if extra else "")
