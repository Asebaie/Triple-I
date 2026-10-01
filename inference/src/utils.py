import glob
import json
import os

_N_THREADS = str(min(os.cpu_count() or 1, 20))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("OMP_NUM_THREADS", _N_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _N_THREADS)
os.environ.setdefault("RAYON_NUM_THREADS", _N_THREADS)

import multiprocessing as mp
import platform
import time

import numpy as np
import pandas as pd
import src.matching as matching_module
from src.lgbm_numpy import load_model
from src.matching import (
    CAT_TO_IDX,
    EMPTY_RECORD,
    FEATURE_NAMES,
    build_idf,
    build_records,
    ce_text_pair,
    features_for_pairs,
)

_RECORDS = None
_IDF = None
_IDS1 = None
_IDS2 = None

START_TIME = time.time()
SANITY_FEATURE = "idf_overlap_min"
SANITY_SAMPLE = 20000
SANITY_MIN_CORRELATION = 0.25


def log(message):
    print(f"[{time.time() - START_TIME:7.1f}s] {message}", flush=True)


def report_environment():
    log(f"python {platform.python_version()} | {platform.platform()} | ядер {os.cpu_count()}")
    for name in ("numpy", "pandas", "pyarrow", "torch", "transformers", "tokenizers"):
        try:
            module = __import__(name)
            log(f"версия {name}: {getattr(module, '__version__', 'неизвестна')}")
        except Exception as error:
            log(f"версия {name}: НЕДОСТУПЕН ({type(error).__name__})")


def _feature_chunk(bounds):
    lo, hi = bounds
    return lo, features_for_pairs(_IDS1[lo:hi], _IDS2[lo:hi], _RECORDS, _IDF)


def compute_features(ids1, ids2, workers):
    global _IDS1, _IDS2
    _IDS1, _IDS2 = ids1, ids2
    total = len(ids1)
    out = np.empty((total, len(FEATURE_NAMES)), dtype=np.float32)
    if workers <= 1 or total < 20000 or not hasattr(os, "fork"):
        return features_for_pairs(ids1, ids2, _RECORDS, _IDF, out=out)
    chunk = (total + workers - 1) // workers
    bounds = [(i, min(i + chunk, total)) for i in range(0, total, chunk)]
    context = mp.get_context("fork")
    with context.Pool(processes=len(bounds)) as pool:
        for lo, block in pool.imap_unordered(_feature_chunk, bounds):
            out[lo : lo + len(block)] = block
    return out


def load_booster(path, n_features, label):
    if not os.path.exists(path):
        log(f"{label}: файл не найден ({path})")
        return None
    try:
        booster = load_model(path)
        if booster.n_features != n_features:
            raise ValueError(f"признаков в модели {booster.n_features}, ожидалось {n_features}")
        log(f"{label}: загружено деревьев {booster.num_trees}, признаков {booster.n_features}")
        return booster
    except Exception as error:
        log(f"{label}: не загрузилось — {error!r}")
        return None


def _tokenize_batches(texts_a, texts_b, tokenizer, max_len, batch_size, order):
    batches = []
    for start in range(0, len(order), batch_size):
        index = order[start : start + batch_size]
        batches.append(
            tokenizer(
                [texts_a[i] for i in index],
                [texts_b[i] for i in index],
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
        )
    return batches


def _model_signature(model_dir, default_max_len):
    max_len = default_max_len
    attr_keys = True
    categories = ()
    weight = 1.0
    config_path = os.path.join(model_dir, "training_config.json")
    if os.path.exists(config_path):
        try:
            saved = json.load(open(config_path, encoding="utf-8"))
            max_len = int(saved.get("max_len", default_max_len))
            attr_keys = bool(saved.get("attr_keys", True))
            categories = tuple(saved.get("categories") or ())
            weight = float(saved.get("weight", 1.0))
        except Exception as error:
            log(f"{os.path.basename(model_dir)}: не читается training_config.json — {error!r}")
    vocab = None
    try:
        vocab = int(json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))["vocab_size"])
    except Exception as error:
        log(f"{os.path.basename(model_dir)}: не читается config.json — {error!r}")
    return vocab, max_len, attr_keys, categories, weight


def run_cross_encoders(model_dirs, text_sets, default_max_len, batch_size, pair_categories=None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        major = torch.cuda.get_device_capability(0)[0]
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        log(f"GPU {torch.cuda.get_device_name(0)} | capability {major} | dtype {dtype}")
    else:
        dtype = torch.float32
        log("GPU недоступна, кросс-энкодер считается на CPU")

    def load_model_safe(path):
        for kwargs in (
            {"dtype": dtype, "attn_implementation": "sdpa"},
            {"torch_dtype": dtype, "attn_implementation": "sdpa"},
            {"dtype": dtype},
            {"torch_dtype": dtype},
            {},
        ):
            try:
                return AutoModelForSequenceClassification.from_pretrained(path, num_labels=1, **kwargs)
            except TypeError:
                continue
        raise RuntimeError(f"не удалось загрузить модель из {path}")

    groups = {}
    for model_dir in model_dirs:
        vocab, model_max_len, attr_keys, restrict, weight = _model_signature(model_dir, default_max_len)
        groups.setdefault((vocab, model_max_len, attr_keys), []).append((model_dir, restrict, weight))
    log(f"Групп токенизации: {len(groups)} на {len(model_dirs)} моделей")

    total = len(text_sets[True][0])
    columns = []
    weights = []
    done = 0
    for (vocab, max_len, attr_keys), members in groups.items():
        texts_a, texts_b = text_sets[attr_keys]
        lengths = np.fromiter((len(texts_a[i]) + len(texts_b[i]) for i in range(total)), dtype=np.int32, count=total)
        full_order = np.argsort(lengths, kind="stable")
        tokenizer = AutoTokenizer.from_pretrained(members[0][0])
        log(
            f"Группа vocab={vocab}, max_len={max_len}, ключи={attr_keys}: {[os.path.basename(d) for d, _, _ in members]}"
        )
        restrictions = {tuple(restrict) for _, restrict, _ in members}
        batch_cache = {}
        for restrict in restrictions:
            order = full_order
            if restrict and pair_categories is not None:
                allowed = [CAT_TO_IDX[c] for c in restrict if c in CAT_TO_IDX]
                order = full_order[np.isin(pair_categories, allowed)[full_order]]
                log(f"  ограничение {restrict}: {len(order):,} пар")
            batch_cache[restrict] = (order, _tokenize_batches(texts_a, texts_b, tokenizer, max_len, batch_size, order))
        for model_dir, restrict, weight in members:
            order, batches = batch_cache[tuple(restrict)]
            if len(order) == 0:
                continue
            model = load_model_safe(model_dir).to(device).eval()
            column = np.full(total, np.nan, dtype=np.float32)
            offset = 0
            with torch.inference_mode():
                for batch_index, batch in enumerate(batches):
                    inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                    logits = model(**inputs).logits.squeeze(-1).float().cpu().numpy()
                    size = len(logits)
                    column[order[offset : offset + size]] = logits
                    offset += size
                    if (batch_index + 1) % 200 == 0:
                        log(f"{os.path.basename(model_dir)}: батч {batch_index + 1}/{len(batches)}")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            columns.append(column)
            weights.append(weight)
            done += 1
            log(f"{os.path.basename(model_dir)} отработала ({done}/{len(model_dirs)}), вес {weight}")
        del batch_cache

    matrix = np.column_stack(columns)
    weights = np.asarray(weights, dtype=np.float32)
    for index in range(matrix.shape[1]):
        covered = ~np.isnan(matrix[:, index])
        values = matrix[covered, index]
        spread = values.std()
        matrix[covered, index] = (values - values.mean()) / (spread if spread > 1e-6 else 1.0)
    present = ~np.isnan(matrix)
    matrix = np.nan_to_num(matrix, nan=0.0)
    denominator = present * weights
    counts = present.sum(axis=1).astype(np.float32)
    combined = (matrix * denominator).sum(axis=1) / np.maximum(denominator.sum(axis=1), 1e-6)
    return combined.astype(np.float32), counts


def predict_pipeline(
    items_path,
    matches_path,
    output_path,
    models_dir="models",
    gbdt_dir=".",
    max_len=192,
    batch_size=512,
    workers=None,
):
    if workers is None:
        workers = min(os.cpu_count() or 1, 16)
    report_environment()

    n_features = len(FEATURE_NAMES)
    stack_booster = load_booster(os.path.join(gbdt_dir, "gbdt_lgb.txt"), n_features + 1, "GBDT со скором энкодера")
    plain_booster = load_booster(os.path.join(gbdt_dir, "gbdt_features_lgb.txt"), n_features, "GBDT без энкодера")

    items_df = pd.read_parquet(items_path)
    log(f"Товаров: {len(items_df):,} | колонки {list(items_df.columns)}")

    global _RECORDS, _IDF
    _RECORDS = build_records(items_df)
    del items_df
    known = sum(1 for record in _RECORDS.values() if record[2] >= 0)
    log(f"Записей: {len(_RECORDS):,} | с распознанной категорией: {known:,}")

    _IDF = build_idf(_RECORDS)
    matches_df = pd.read_parquet(matches_path)
    ids1 = matches_df["id1"].to_numpy()
    ids2 = matches_df["id2"].to_numpy()
    del matches_df
    log(f"Пар: {len(ids1):,}")

    features = compute_features(ids1, ids2, workers)
    sanity_column = features[:, FEATURE_NAMES.index(SANITY_FEATURE)]
    log(f"Признаки: {features.shape} | {SANITY_FEATURE}: mean={sanity_column.mean():.4f}")

    def build_text_set(records):
        left_texts, right_texts = [], []
        for index in range(len(ids1)):
            left, right = ce_text_pair(records.get(ids1[index], EMPTY_RECORD), records.get(ids2[index], EMPTY_RECORD))
            left_texts.append(left)
            right_texts.append(right)
        return left_texts, right_texts

    model_dirs = sorted(glob.glob(os.path.join(models_dir, "ce*")))
    ce_scores = None
    if model_dirs:
        try:
            needed = {_model_signature(d, max_len)[2] for d in model_dirs}
            text_sets = {True: build_text_set(_RECORDS)}
            log(f"Тексты собраны, пример: {text_sets[True][0][0][:120]!r}")
            if False in needed:
                matching_module.USE_ATTR_KEYS = False
                keyless_records = build_records(pd.read_parquet(items_path))
                matching_module.USE_ATTR_KEYS = True
                text_sets[False] = build_text_set(keyless_records)
                del keyless_records
                log(f"Тексты без ключей собраны, пример: {text_sets[False][0][0][:120]!r}")
            pair_cats = np.array([_RECORDS.get(i, EMPTY_RECORD)[2] for i in ids1], dtype=np.int32)
            ce_scores, counts = run_cross_encoders(model_dirs, text_sets, max_len, batch_size, pair_cats)
            del text_sets
            if (counts == 0).any():
                log(f"ВНИМАНИЕ: {(counts == 0).sum():,} пар не покрыты ни одной моделью")
            log(f"Моделей на пару: минимум {int(counts.min())}, максимум {int(counts.max())}")
        except Exception as error:
            log(f"Кросс-энкодер упал: {error!r}")
    else:
        log(f"Модели кросс-энкодера не найдены в {models_dir}")

    plain_predictions = None
    sanity_index = None
    if plain_booster is not None:
        started = time.time()
        if ce_scores is not None and len(features) > SANITY_SAMPLE:
            sanity_index = np.linspace(0, len(features) - 1, SANITY_SAMPLE).astype(np.int64)
            plain_predictions = plain_booster.predict(features[sanity_index])
        else:
            plain_predictions = plain_booster.predict(features)
        log(f"GBDT без энкодера отработал за {time.time() - started:.1f} с, строк {len(plain_predictions):,}")

    if ce_scores is not None:
        if plain_predictions is not None:
            reference = plain_predictions
            subject = ce_scores if sanity_index is None else ce_scores[sanity_index]
            reference_name = "GBDT без энкодера"
        else:
            reference = sanity_column
            subject = ce_scores
            reference_name = SANITY_FEATURE
        correlation = float(np.corrcoef(subject, reference)[0, 1])
        log(f"Скор энкодера: mean={ce_scores.mean():+.3f} std={ce_scores.std():.3f}")
        log(f"Корреляция скора энкодера с {reference_name}: {correlation:.3f}")
        if not np.isfinite(correlation) or correlation < SANITY_MIN_CORRELATION:
            log(f"Корреляция ниже порога {SANITY_MIN_CORRELATION}, скор энкодера признан недостоверным")
            ce_scores = None

    predictions = None
    if ce_scores is not None and stack_booster is not None:
        matrix = np.hstack([features, ce_scores.reshape(-1, 1)]).astype(np.float32)
        started = time.time()
        predictions = stack_booster.predict(matrix)
        log(f"Ответ: GBDT поверх скора кросс-энкодера, за {time.time() - started:.1f} с")
        del matrix
    if predictions is None and ce_scores is not None:
        predictions = 1.0 / (1.0 + np.exp(-ce_scores.astype(np.float64)))
        log("Ответ: сигмоида скора кросс-энкодера")
    if predictions is None and plain_booster is not None:
        predictions = plain_booster.predict(features)
        log("Ответ: GBDT только на ручных признаках")
    if predictions is None:
        predictions = sanity_column.astype(np.float64)
        log(f"Ответ: сырая фича {SANITY_FEATURE}")

    predictions = np.asarray(predictions, dtype=np.float64)
    if not np.isfinite(predictions).all():
        log("В ответе есть нечисловые значения, заменяю их")
        predictions = np.nan_to_num(predictions, nan=0.0, posinf=1.0, neginf=0.0)

    log(
        f"predict: mean={predictions.mean():.4f} std={predictions.std():.4f} уникальных={len(np.unique(predictions)):,}"
    )
    pd.DataFrame({"id1": ids1, "id2": ids2, "predict": predictions}).to_csv(output_path, index=False)
    log(f"Сохранено {len(ids1):,} строк в {output_path}")
