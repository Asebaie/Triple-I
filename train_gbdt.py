import argparse
import gc
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import lightgbm as lgb

from core.lgbm_numpy import load_model
from core.logging_utils import StageTimer, setup_logger
from core.matching import FEATURE_NAMES, build_idf, build_records, features_for_pairs
from core.metrics import macro_pr_auc, macro_pr_auc_at_prevalence

PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 96,
    "min_data_in_leaf": 60,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "verbose": -1,
    "num_threads": os.cpu_count(),
    "seed": 42,
}


def group_split(ids1, ids2, n_splits, seed):
    codes, _ = pd.factorize(np.concatenate([ids1, ids2]))
    left = codes[: len(ids1)]
    right = codes[len(ids1) :]
    parent = np.arange(codes.max() + 1, dtype=np.int64)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in zip(left, right):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    groups = np.array([find(x) for x in left], dtype=np.int64)
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    mapping = {g: i % n_splits for i, g in enumerate(unique_groups)}
    return np.array([mapping[g] for g in groups], dtype=np.int32)


def verify_numpy_predictor(path, matrix, logger):
    booster = load_model(path)
    reference = lgb.Booster(model_file=path)
    sample = matrix[: min(20000, len(matrix))]
    diff = np.abs(booster.predict(sample) - reference.predict(sample)).max()
    logger.info(f"Сверка numpy-предсказателя с lightgbm: max расхождение = {diff:.3e}")
    if diff > 1e-6:
        raise SystemExit("numpy-предсказатель расходится с lightgbm, сабмит собирать нельзя")
    logger.info(f"numpy-предсказатель исправен, деревьев {booster.num_trees}, признаков {booster.n_features}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="data/items_human.parquet")
    parser.add_argument("--oof_dir", default="artifacts")
    parser.add_argument("--out_dir", default="artifacts")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--num_rounds", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logger, _ = setup_logger("train_gbdt", args.log_dir)
    logger.info(f"Аргументы: {vars(args)}")

    oof_files = sorted(glob.glob(os.path.join(args.oof_dir, "oof_fold*.parquet")))
    if not oof_files:
        raise SystemExit(f"Не найдены OOF файлы в {args.oof_dir}. Сначала запусти train_ce.py")
    oof = pd.concat([pd.read_parquet(f) for f in oof_files], ignore_index=True)
    logger.info(f"OOF файлы: {oof_files} | пар: {len(oof):,}")

    with StageTimer(logger, "Построение записей товаров и IDF"):
        items_df = pd.read_parquet(args.items)
        records = build_records(items_df, log_every=200000, logger=logger)
        del items_df
        gc.collect()
        idf = build_idf(records)

    ids1 = oof["id1"].to_numpy()
    ids2 = oof["id2"].to_numpy()
    targets = oof["target"].to_numpy(dtype=np.float32)
    categories = oof["category"].to_numpy(dtype=np.int32)
    ce_score = oof["ce_score"].to_numpy(dtype=np.float32)

    with StageTimer(logger, "Расчёт ручных признаков"):
        features = features_for_pairs(ids1, ids2, records, idf, log_every=50000, logger=logger)
    del records
    gc.collect()

    matrix = np.hstack([features, ce_score.reshape(-1, 1)]).astype(np.float32)
    feature_names = FEATURE_NAMES + ["ce_score"]
    logger.info(f"Матрица признаков: {matrix.shape}")

    base_score, _ = macro_pr_auc(categories, targets, ce_score)
    logger.info(f"Базовый macro PR-AUC только по кросс-энкодеру: {base_score:.5f}")

    splits = group_split(ids1, ids2, args.splits, args.seed)
    oof_pred = np.zeros(len(targets), dtype=np.float32)
    best_iters = []
    for split in range(args.splits):
        train_mask = splits != split
        params = dict(PARAMS, seed=args.seed + split)
        booster = lgb.train(
            params,
            lgb.Dataset(matrix[train_mask], label=targets[train_mask]),
            num_boost_round=args.num_rounds,
            valid_sets=[lgb.Dataset(matrix[~train_mask], label=targets[~train_mask])],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof_pred[~train_mask] = booster.predict(matrix[~train_mask], num_iteration=booster.best_iteration)
        best_iters.append(booster.best_iteration)
        split_score, _ = macro_pr_auc(categories[~train_mask], targets[~train_mask], oof_pred[~train_mask])
        logger.info(f"Сплит {split}: macro PR-AUC = {split_score:.5f} | итераций {booster.best_iteration}")

    stack_score, per_category = macro_pr_auc(categories, targets, oof_pred)
    logger.info(f"Стек (кросс-энкодер + GBDT) macro PR-AUC = {stack_score:.5f}")
    logger.info(f"Прирост относительно чистого кросс-энкодера: {stack_score - base_score:+.5f}")
    base_low, _ = macro_pr_auc_at_prevalence(categories, targets, ce_score)
    stack_low, spread = macro_pr_auc_at_prevalence(categories, targets, oof_pred)
    logger.info(
        f"ОЖИДАНИЕ ДЛЯ ЛИДЕРБОРДА при доле позитивов 5%: энкодер {base_low:.4f}, стек {stack_low:.4f} (разброс {spread:.4f})"
    )
    for cat_idx, cat_score in sorted(per_category.items()):
        logger.info(f"    категория {cat_idx}: PR-AUC = {cat_score:.4f}")

    rounds = max(int(np.mean(best_iters) * 1.1), 100)
    logger.info(f"Финальное обучение на всех OOF, итераций: {rounds}")
    booster = lgb.train(dict(PARAMS, seed=args.seed), lgb.Dataset(matrix, label=targets), num_boost_round=rounds)
    lgb_path = os.path.join(args.out_dir, "gbdt_lgb.txt")
    booster.save_model(lgb_path)
    logger.info(f"Модель сохранена: {lgb_path}")

    importance = sorted(zip(feature_names, booster.feature_importance("gain")), key=lambda x: -x[1])
    for name, gain in importance[:20]:
        logger.info(f"    важность {name}: {gain:.0f}")

    verify_numpy_predictor(lgb_path, matrix, logger)

    for stale in ("gbdt_sklearn.joblib",):
        path = os.path.join(args.out_dir, stale)
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Удалён устаревший артефакт: {stale}")

    with open(os.path.join(args.out_dir, "stack_report.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"ce_only={base_score:.5f}\nstack={stack_score:.5f}\nrounds={rounds}\n")
    logger.info("Готово. Дальше запускай make_submission.py")


if __name__ == "__main__":
    main()
