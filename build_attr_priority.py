import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.logging_utils import StageTimer, setup_logger
from core.matching import CATEGORIES, _parse_attributes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="data/items_human.parquet")
    parser.add_argument("--matches", default="data/matches.parquet")
    parser.add_argument("--out", default="ecup/attr_priority.py")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--min_count", type=int, default=150)
    args = parser.parse_args()

    logger, _ = setup_logger("attr_priority", args.log_dir)

    with StageTimer(logger, "Разбор атрибутов"):
        items = pd.read_parquet(args.items)
        category_of = dict(zip(items["id"], items["category"]))
        attributes = {row.id: dict(_parse_attributes(row.attributes)) for row in items.itertuples()}
        del items
        matches = pd.read_parquet(args.matches)

    stats = defaultdict(lambda: [0, 0.0, 0, 0.0])
    with StageTimer(logger, "Сбор статистики по парам"):
        for id1, id2, target in zip(matches.id1.to_numpy(), matches.id2.to_numpy(), matches.target.to_numpy()):
            first, second = attributes.get(id1), attributes.get(id2)
            if not first or not second:
                continue
            category = category_of.get(id1)
            for key in first.keys() & second.keys():
                bucket = stats[(category, key)]
                if first[key] == second[key]:
                    bucket[0] += 1
                    bucket[1] += target
                else:
                    bucket[2] += 1
                    bucket[3] += target

    pairs_in_category = matches.id1.map(category_of).value_counts().to_dict()
    ranked = {}
    for category in CATEGORIES:
        rows = []
        total = max(pairs_in_category.get(category, 1), 1)
        for (cat, key), (n_equal, pos_equal, n_diff, pos_diff) in stats.items():
            if cat != category or n_equal < args.min_count or n_diff < args.min_count:
                continue
            gap = abs(pos_equal / n_equal - pos_diff / n_diff)
            coverage = (n_equal + n_diff) / total
            rows.append((key, gap * np.sqrt(coverage)))
        rows.sort(key=lambda item: -item[1])
        ranked[category] = [key for key, _ in rows[: args.top]]
        logger.info(f"{category}: отобрано {len(ranked[category])} атрибутов")

    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("CATEGORY_ATTR_ORDER = {\n")
        for category, keys in ranked.items():
            handle.write(f"    {category!r}: [\n")
            for key in keys:
                handle.write(f"        {key!r},\n")
            handle.write("    ],\n")
        handle.write("}\n\n")
        handle.write("CATEGORY_ATTR_RANK = {\n")
        handle.write("    category: {key: index for index, key in enumerate(keys)}\n")
        handle.write("    for category, keys in CATEGORY_ATTR_ORDER.items()\n")
        handle.write("}\n")
    logger.info(f"Таблица сохранена: {args.out}")


if __name__ == "__main__":
    main()
