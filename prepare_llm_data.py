import argparse
import gc
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.logging_utils import Heartbeat, StageTimer, setup_logger
from core.matching import CATEGORIES, N_CATEGORIES, build_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="data/items.parquet")
    parser.add_argument("--matches_llm", default="data/matches_llm.parquet")
    parser.add_argument("--out", default="data/llm_train.parquet")
    parser.add_argument("--n_pairs", type=int, default=1500000)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=250000)
    parser.add_argument("--read_batch", type=int, default=200000)
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger, _ = setup_logger("prepare_llm", args.log_dir)
    logger.info(f"Аргументы: {vars(args)}")

    with StageTimer(logger, "Чтение и подвыборка LLM-пар"):
        matches = pd.read_parquet(args.matches_llm, columns=["id1", "id2", "target"])
        logger.info(f"Всего LLM-пар: {len(matches):,}")
        if args.n_pairs and len(matches) > args.n_pairs:
            matches = matches.sample(args.n_pairs, random_state=args.seed)
        matches = matches.reset_index(drop=True)
        if args.shards > 1:
            matches = matches.iloc[args.shard :: args.shards].reset_index(drop=True)
            stem, ext = os.path.splitext(args.out)
            args.out = f"{stem}_{args.shard}{ext}"
            logger.info(f"Часть {args.shard + 1} из {args.shards}: {len(matches):,} пар, файл {args.out}")
        ids1 = matches["id1"].to_numpy()
        ids2 = matches["id2"].to_numpy()
        targets = matches["target"].to_numpy(dtype=np.float32)
        del matches
        gc.collect()
        needed = np.unique(np.concatenate([ids1, ids2]))
        logger.info(f"Отобрано пар: {len(ids1):,} | уникальных товаров: {len(needed):,}")

    texts = {}
    categories = {}
    with StageTimer(logger, "Потоковое чтение items.parquet"):
        parquet_file = pq.ParquetFile(args.items)
        total_rows = parquet_file.metadata.num_rows
        heartbeat = Heartbeat(logger, total_rows, "Просмотрено товаров", every_seconds=20.0)
        for batch in parquet_file.iter_batches(
            batch_size=args.read_batch, columns=["id", "name", "attributes", "category"]
        ):
            batch_ids = batch.column("id").to_numpy()
            mask = np.isin(batch_ids, needed, assume_unique=False)
            heartbeat.update(len(batch_ids))
            if not mask.any():
                continue
            selected = np.flatnonzero(mask)
            names = batch.column("name").to_pylist()
            attributes = batch.column("attributes").to_pylist()
            cats = batch.column("category").to_pylist()
            for i in selected:
                record = build_record(names[i], attributes[i], cats[i])
                texts[batch_ids[i]] = f"{record[0]} ; {record[1]}"
                categories[batch_ids[i]] = record[2]
            del names, attributes, cats
        heartbeat.finish()
    logger.info(f"Собрано текстов: {len(texts):,}")
    del needed
    gc.collect()

    schema = pa.schema(
        [("text1", pa.large_string()), ("text2", pa.large_string()), ("target", pa.float32()), ("category", pa.int8())]
    )
    written = 0
    with StageTimer(logger, "Запись обучающего файла"):
        writer = pq.ParquetWriter(args.out, schema, compression="zstd")
        heartbeat = Heartbeat(logger, len(ids1), "Записано пар", every_seconds=20.0)
        for start in range(0, len(ids1), args.chunk):
            stop = min(start + args.chunk, len(ids1))
            block_text1 = []
            block_text2 = []
            block_target = []
            block_cat = []
            for i in range(start, stop):
                left = texts.get(ids1[i])
                right = texts.get(ids2[i])
                if left is None or right is None:
                    continue
                cat_idx = categories.get(ids1[i], -1)
                prefix = CATEGORIES[cat_idx] if 0 <= cat_idx < N_CATEGORIES else "товар"
                block_text1.append(f"{prefix} ; {left}")
                block_text2.append(right)
                block_target.append(targets[i])
                block_cat.append(cat_idx)
            if block_text1:
                writer.write_table(
                    pa.Table.from_arrays(
                        [
                            pa.array(block_text1, type=pa.large_string()),
                            pa.array(block_text2, type=pa.large_string()),
                            pa.array(block_target, type=pa.float32()),
                            pa.array(block_cat, type=pa.int8()),
                        ],
                        schema=schema,
                    )
                )
                written += len(block_text1)
            heartbeat.update(stop - start)
        heartbeat.finish()
        writer.close()

    logger.info(f"Готово: {args.out} | пар записано {written:,} | размер {os.path.getsize(args.out) / 1e6:.1f} МБ")


if __name__ == "__main__":
    main()
