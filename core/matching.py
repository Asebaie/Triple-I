import json
import re
import zlib
from difflib import SequenceMatcher

import numpy as np

try:
    from ecup.attr_priority import CATEGORY_ATTR_RANK
except ImportError:
    try:
        from src.attr_priority import CATEGORY_ATTR_RANK
    except ImportError:
        CATEGORY_ATTR_RANK = {}

CATEGORIES = [
    "Автотовары",
    "Аптека",
    "Бытовая техника",
    "Бытовая химия",
    "Галантерея и аксессуары",
    "Детские товары",
    "Дом и сад",
    "Канцелярские товары",
    "Красота и гигиена",
    "Мебель",
    "Музыкальные инструменты",
    "Обувь",
    "Одежда",
    "Продукты питания",
    "Спорт и отдых",
    "Строительство и ремонт",
    "Товары для животных",
    "Хобби и творчество",
    "Электроника",
    "Ювелирные изделия",
]
CAT_TO_IDX = {c: i for i, c in enumerate(CATEGORIES)}
N_CATEGORIES = len(CATEGORIES)

MAX_NAME_CHARS = 160
MAX_KV_CHARS = 640
MAX_VALUE_CHARS = 48
MAX_ATTRS = 32
USE_ATTR_KEYS = True

ATTR_PRIORITY = (
    "бренд",
    "brand",
    "торговая марка",
    "модель",
    "серия",
    "артикул",
    "партномер",
    "oem",
    "тип",
    "вид",
    "цвет",
    "размер",
    "объем",
    "объём",
    "вес",
    "мощность",
    "количество",
    "единиц",
    "материал",
    "состав",
    "вкус",
    "аромат",
    "назначение",
    "пол",
    "возраст",
)

ATTR_TEXT_BLACKLIST = (
    "валюта",
    "ндс",
    "предупреждени",
    "код товара",
)

EMPTY_VALUES = {"нет", "-", "--", "none", "null", "не указано", "отсутствует", "n/a", "нд", "0"}

_re_ws = re.compile(r"\s+")
_re_word = re.compile(r"[a-zа-я0-9]+")
_re_num = re.compile(r"\d+(?:[.,]\d+)?")
_re_alnum = re.compile(r"[a-z0-9]+")
_re_keep = re.compile(r"[^a-zа-я0-9 .,/=|-]+")

IDF_BITS = 20
IDF_SIZE = 1 << IDF_BITS
IDF_MASK = IDF_SIZE - 1

EMPTY_RECORD = ("", "", -1, "", "", "", "")


def normalize(text):
    text = text.lower().replace("ё", "е").replace(" ", " ")
    text = _re_keep.sub(" ", text)
    return _re_ws.sub(" ", text).strip()


def _attr_rank(key, category=None):
    learned = CATEGORY_ATTR_RANK.get(category) if category else None
    if learned is not None and key in learned:
        return learned[key]
    for i, token in enumerate(ATTR_PRIORITY):
        if token in key:
            return len(CATEGORY_ATTR_RANK.get(category, ())) + i
    return len(CATEGORY_ATTR_RANK.get(category, ())) + len(ATTR_PRIORITY)


def _parse_attributes(raw):
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    out = []
    for key, value in data.items():
        if isinstance(value, (list, tuple)):
            value = " ".join(str(x) for x in value)
        key_norm = normalize(str(key))
        value_norm = normalize(str(value))
        if not key_norm or not value_norm or value_norm in EMPTY_VALUES:
            continue
        out.append((key_norm, value_norm[:MAX_VALUE_CHARS]))
    return out


def build_record(name, attributes, category):
    name_norm = normalize(str(name))[:MAX_NAME_CHARS]
    attrs = _parse_attributes(attributes)
    attrs.sort(key=lambda kv: _attr_rank(kv[0], category))

    brand = model = article = color = ""
    parts = []
    used = 0
    for key, value in attrs[:MAX_ATTRS]:
        if not brand and ("бренд" in key or "brand" in key or "торговая марка" in key):
            brand = value
        if not model and key.startswith("модель"):
            model = value
        if not article and ("артикул" in key or "партномер" in key):
            article = "".join(_re_alnum.findall(value))
        if not color and "цвет" in key:
            color = value
        if any(bad in key for bad in ATTR_TEXT_BLACKLIST):
            continue
        piece = (key + "=" + value) if USE_ATTR_KEYS else value
        if used + len(piece) > MAX_KV_CHARS:
            continue
        parts.append(piece)
        used += len(piece) + 1

    return (name_norm, "|".join(parts), CAT_TO_IDX.get(category, -1), brand, model, article, color)


def build_records(items_df, log_every=0, logger=None):
    records = {}
    ids = items_df["id"].to_numpy()
    names = items_df["name"].to_numpy()
    attributes = items_df["attributes"].to_numpy()
    categories = items_df["category"].to_numpy()
    total = len(ids)
    for i in range(total):
        records[ids[i]] = build_record(names[i], attributes[i], categories[i])
        if log_every and logger is not None and (i + 1) % log_every == 0:
            logger.info(f"Построено записей: {i + 1:,}/{total:,}")
    return records


def category_name(record):
    idx = record[2]
    return CATEGORIES[idx] if 0 <= idx < N_CATEGORIES else "товар"


def ce_text_pair(record1, record2):
    left = f"{category_name(record1)} ; {record1[0]} ; {record1[1]}"
    right = f"{record2[0]} ; {record2[1]}"
    return left, right


def _hash_word(word):
    return zlib.crc32(word.encode("utf-8")) & IDF_MASK


def build_idf(records, values=None):
    counts = np.zeros(IDF_SIZE, dtype=np.int32)
    source = values if values is not None else records.values()
    n_docs = 0
    for record in source:
        n_docs += 1
        for word in set(_re_word.findall(record[0])):
            counts[_hash_word(word)] += 1
    idf = np.log((n_docs + 1.0) / (counts + 1.0)).astype(np.float32)
    return idf


def _codes(words):
    return {w for w in words if len(w) >= 4 and any(c.isdigit() for c in w) and not w.isdigit()}


def _trigrams(text):
    return {text[i : i + 3] for i in range(max(len(text) - 2, 0))}


def _match_state(a, b):
    if not a or not b:
        return 0.5
    return 1.0 if a == b else 0.0


def _kv_sets(kv):
    if not kv:
        return {}, set()
    mapping = {}
    values = set()
    for index, piece in enumerate(kv.split("|")):
        pos = piece.find("=")
        if pos <= 0:
            mapping[f"#{index}"] = piece
            values.add(piece)
            continue
        key = piece[:pos]
        value = piece[pos + 1 :]
        mapping[key] = value
        values.add(value)
    return mapping, values


FEATURE_NAMES = [
    "len1",
    "len2",
    "len_ratio",
    "len_diff",
    "nw1",
    "nw2",
    "w_inter",
    "w_jaccard",
    "w_dice",
    "w_overlap_min",
    "first_word_eq",
    "prefix_ratio",
    "tri_jaccard",
    "tri_overlap_min",
    "seq_ratio",
    "name_eq",
    "tokset_eq",
    "num1",
    "num2",
    "num_inter",
    "num_jaccard",
    "num_conflict",
    "num_overlap_min",
    "code1",
    "code2",
    "code_inter",
    "code_jaccard",
    "code_conflict",
    "idf_inter",
    "idf_jaccard",
    "idf_overlap_min",
    "idf_max_missed",
    "brand_state",
    "model_state",
    "article_state",
    "color_state",
    "nkv1",
    "nkv2",
    "kv_common_keys",
    "kv_equal",
    "kv_conflict",
    "kv_equal_ratio",
    "kv_conflict_ratio",
    "val_inter",
    "val_jaccard",
    "kv_len1",
    "kv_len2",
    "category",
] + [f"cat_{i}" for i in range(N_CATEGORIES)]
N_FEATURES = len(FEATURE_NAMES)
CATEGORY_FEATURE_INDEX = FEATURE_NAMES.index("category")
CATEGORY_ONEHOT_START = CATEGORY_FEATURE_INDEX + 1


def pair_features(record1, record2, idf, out=None):
    name1, kv1, cat1, brand1, model1, art1, color1 = record1
    name2, kv2, _, brand2, model2, art2, color2 = record2

    len1, len2 = len(name1), len(name2)
    len_ratio = min(len1, len2) / max(len1, len2, 1)

    words1 = _re_word.findall(name1)
    words2 = _re_word.findall(name2)
    set1, set2 = set(words1), set(words2)
    inter = set1 & set2
    union = set1 | set2
    n_inter, n_union = len(inter), len(union)
    min_words = max(min(len(set1), len(set2)), 1)

    first_word_eq = 1.0 if words1 and words2 and words1[0] == words2[0] else 0.0
    prefix = 0
    for a, b in zip(name1, name2):
        if a != b:
            break
        prefix += 1
    prefix_ratio = prefix / max(min(len1, len2), 1)

    tri1, tri2 = _trigrams(name1), _trigrams(name2)
    tri_inter = len(tri1 & tri2)
    tri_union = len(tri1 | tri2)
    tri_min = max(min(len(tri1), len(tri2)), 1)

    seq_ratio = SequenceMatcher(None, name1[:120], name2[:120]).quick_ratio() if len1 and len2 else 0.0

    nums1 = set(_re_num.findall(name1 + " " + kv1))
    nums2 = set(_re_num.findall(name2 + " " + kv2))
    num_inter = len(nums1 & nums2)
    num_union = len(nums1 | nums2)
    num_min = max(min(len(nums1), len(nums2)), 1)

    codes1, codes2 = _codes(set1), _codes(set2)
    code_inter = len(codes1 & codes2)
    code_union = len(codes1 | codes2)

    idf_inter = 0.0
    idf_union = 0.0
    idf_max_missed = 0.0
    for word in union:
        weight = float(idf[_hash_word(word)])
        idf_union += weight
        if word in inter:
            idf_inter += weight
        elif weight > idf_max_missed:
            idf_max_missed = weight
    idf_self1 = sum(float(idf[_hash_word(w)]) for w in set1)
    idf_self2 = sum(float(idf[_hash_word(w)]) for w in set2)
    idf_min = max(min(idf_self1, idf_self2), 1e-6)

    map1, values1 = _kv_sets(kv1)
    map2, values2 = _kv_sets(kv2)
    common_keys = set(map1) & set(map2)
    kv_equal = sum(1 for k in common_keys if map1[k] == map2[k])
    kv_conflict = len(common_keys) - kv_equal
    val_inter = len(values1 & values2)
    val_union = len(values1 | values2)

    if out is None:
        out = np.zeros(N_FEATURES, dtype=np.float32)
    else:
        out[CATEGORY_ONEHOT_START:] = 0.0
    if 0 <= cat1 < N_CATEGORIES:
        out[CATEGORY_ONEHOT_START + cat1] = 1.0
    out[:CATEGORY_ONEHOT_START] = [
        len1,
        len2,
        len_ratio,
        abs(len1 - len2),
        len(set1),
        len(set2),
        n_inter,
        n_inter / n_union if n_union else 0.0,
        2.0 * n_inter / max(len(set1) + len(set2), 1),
        n_inter / min_words,
        first_word_eq,
        prefix_ratio,
        tri_inter / tri_union if tri_union else 0.0,
        tri_inter / tri_min,
        seq_ratio,
        1.0 if name1 == name2 and name1 else 0.0,
        1.0 if set1 == set2 and set1 else 0.0,
        len(nums1),
        len(nums2),
        num_inter,
        num_inter / num_union if num_union else 1.0,
        1.0 if nums1 and nums2 and num_inter == 0 else 0.0,
        num_inter / num_min,
        len(codes1),
        len(codes2),
        code_inter,
        code_inter / code_union if code_union else 1.0,
        1.0 if codes1 and codes2 and code_inter == 0 else 0.0,
        idf_inter,
        idf_inter / idf_union if idf_union else 0.0,
        idf_inter / idf_min,
        idf_max_missed,
        _match_state(brand1, brand2),
        _match_state(model1, model2),
        _match_state(art1, art2),
        _match_state(color1, color2),
        len(map1),
        len(map2),
        len(common_keys),
        kv_equal,
        kv_conflict,
        kv_equal / max(len(common_keys), 1),
        kv_conflict / max(len(common_keys), 1),
        val_inter,
        val_inter / val_union if val_union else 0.0,
        len(kv1),
        len(kv2),
        cat1,
    ]
    return out


def features_for_pairs(ids1, ids2, records, idf, out=None, log_every=0, logger=None):
    total = len(ids1)
    if out is None:
        out = np.empty((total, N_FEATURES), dtype=np.float32)
    out[:] = 0.0
    for i in range(total):
        record1 = records.get(ids1[i], EMPTY_RECORD)
        record2 = records.get(ids2[i], EMPTY_RECORD)
        pair_features(record1, record2, idf, out=out[i])
        if log_every and logger is not None and (i + 1) % log_every == 0:
            logger.info(f"Посчитано фич для пар: {i + 1:,}/{total:,}")
    return out


def pair_categories(ids1, records):
    return np.array([records.get(i, EMPTY_RECORD)[2] for i in ids1], dtype=np.int32)
