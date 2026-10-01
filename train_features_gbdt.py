import argparse
import gc
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
from core.matching import CATEGORY_FEATURE_INDEX, FEATURE_NAMES, build_idf, build_records, features_for_pairs
from core.metrics import macro_pr_auc, macro_pr_auc_at_prevalence
from train_gbdt import group_split, verify_numpy_predictor

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="data/items_human.parquet")
    parser.add_argument("--matches", default="data/matches.parquet")
    parser.add_argument("--out_dir", default="artifacts")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--splits", type=int, default=5)
    args = parser.parse_args()

    logger, _ = setup_logger("train_features_gbdt", args.log_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    with StageTimer(logger, "Построение записей и признаков"):
        items_df = pd.read_parquet(args.items)
        records = build_records(items_df, log_every=200000, logger=logger)
        del items_df
        gc.collect()
        idf = build_idf(records)
        matches = pd.read_parquet(args.matches)
        ids1 = matches["id1"].to_numpy()
        ids2 = matches["id2"].to_numpy()
        targets = matches["target"].to_numpy(dtype=np.float32)
        del matches
        gc.collect()
        features = features_for_pairs(ids1, ids2, records, idf, log_every=100000, logger=logger)
        categories = features[:, CATEGORY_FEATURE_INDEX].astype(np.int32)
        del records
        gc.collect()

    logger.info(f"Матрица признаков: {features.shape}")
    splits = group_split(ids1, ids2, args.splits, 42)
    oof = np.zeros(len(targets), dtype=np.float32)
    best_iters = []
    for split in range(args.splits):
        train_mask = splits != split
        booster = lgb.train(
            PARAMS,
            lgb.Dataset(features[train_mask], label=targets[train_mask]),
            num_boost_round=3000,
            valid_sets=[lgb.Dataset(features[~train_mask], label=targets[~train_mask])],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof[~train_mask] = booster.predict(features[~train_mask], num_iteration=booster.best_iteration)
        best_iters.append(booster.best_iteration)
        score, _ = macro_pr_auc(categories[~train_mask], targets[~train_mask], oof[~train_mask])
        logger.info(f"Сплит {split}: macro PR-AUC = {score:.5f} | итераций {booster.best_iteration}")

    score, _ = macro_pr_auc(categories, targets, oof)
    logger.info(f"GBDT только на ручных признаках, честный OOF: macro PR-AUC = {score:.5f}")
    low, spread = macro_pr_auc_at_prevalence(categories, targets, oof)
    logger.info(f"ОЖИДАНИЕ ДЛЯ ЛИДЕРБОРДА при доле позитивов 5%: {low:.4f} (разброс {spread:.4f})")

    rounds = max(int(np.mean(best_iters) * 1.1), 100)
    booster = lgb.train(
        PARAMS,
        lgb.Dataset(features, label=targets),
        num_boost_round=rounds,
    )
    lgb_path = os.path.join(args.out_dir, "gbdt_features_lgb.txt")
    booster.save_model(lgb_path)
    logger.info(f"Сохранено: {lgb_path} | итераций {rounds}")

    verify_numpy_predictor(lgb_path, features, logger)
    logger.info(f"ОЖИДАЕМЫЙ СКОР ЭТОГО ЗОНДА НА ПЛАТФОРМЕ: около {low:.3f}")


if __name__ == "__main__":
    main()
