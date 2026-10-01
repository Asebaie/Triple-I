import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from core.matching import CAT_TO_IDX
from core.metrics import macro_pr_auc


def load(oof_dir, name):
    files = sorted(Path(oof_dir).glob(f"{name}_oof_fold*.parquet"))
    if not files:
        raise SystemExit(f"Нет OOF-файлов для {name} в {oof_dir}")
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return frame[["id1", "id2", "category", "target", "ce_score"]].rename(columns={"ce_score": name})


def zscore(values):
    return (values - np.nanmean(values)) / np.nanstd(values)


def search_general(frame, general, top):
    cats, target = frame["category"].to_numpy(), frame["target"].to_numpy()
    rows = []
    for size in range(1, len(general) + 1):
        for combo in itertools.combinations(general, size):
            score = macro_pr_auc(cats, target, frame[list(combo)].mean(axis=1).to_numpy())[0]
            rows.append((score, size, " + ".join(combo)))
    rows.sort(reverse=True)
    print(f"\nЛучшие ансамбли общих моделей ({len(frame):,} пар)")
    for score, size, name in rows[:top]:
        print(f"  {score:.5f}  [{size}]  {name}")
    print("Лучший состав на каждое число моделей")
    for size in range(1, len(general) + 1):
        score, _, name = max(row for row in rows if row[1] == size)
        print(f"  {size}: {score:.5f}  {name}")


def crossfit_specialists(frame, base_models, specialists, hard, seed, parts):
    hard_ids = [CAT_TO_IDX[name] for name in hard]
    subset = frame[frame["category"].isin(hard_ids)].dropna(subset=specialists).reset_index(drop=True)
    cats, target = subset["category"].to_numpy(), subset["target"].to_numpy()
    base = subset[base_models].mean(axis=1).to_numpy()
    first, second = subset[specialists[0]].to_numpy(), subset[specialists[1]].to_numpy()
    grid = [(a, b) for a in np.arange(0, 0.65, 0.05) for b in np.arange(0, 0.45, 0.05) if a + b <= 0.7 + 1e-9]
    split = np.random.default_rng(seed).permutation(len(subset)) % parts

    def mean_ap(mask, prediction):
        return np.mean(
            [average_precision_score(target[mask & (cats == c)], prediction[mask & (cats == c)]) for c in hard_ids]
        )

    blended, chosen = base.copy(), []
    for part in range(parts):
        train, test = split != part, split == part
        scores = [mean_ap(train, (1 - a - b) * base + a * first + b * second) for a, b in grid]
        a, b = grid[int(np.argmax(scores))]
        chosen.append((a, b))
        blended[test] = (1 - a - b) * base[test] + a * first[test] + b * second[test]

    everything = np.ones(len(subset), dtype=bool)
    before, after = mean_ap(everything, base), mean_ap(everything, blended)
    weights = np.array(chosen)
    share_first, share_second = weights[:, 0].mean(), weights[:, 1].mean()
    print(f"\nСпециалисты на {', '.join(hard)} ({len(subset):,} пар, база {' + '.join(base_models)})")
    print(f"  средний PR-AUC трёх категорий: {before:.5f} -> {after:.5f} ({after - before:+.5f})")
    print(f"  вклад в макро по 20 категориям: {(after - before) * len(hard) / 20:+.5f}")
    print(f"  доля {specialists[0]}: {share_first:.2f} ± {weights[:, 0].std():.2f}")
    print(f"  доля {specialists[1]}: {share_second:.2f} ± {weights[:, 1].std():.2f}")

    share_first, share_second = round(share_first * 10) / 10, round(share_second * 10) / 10
    general_share = 1 - share_first - share_second
    scale = len(base_models) / general_share
    print(
        f"  после округления доли {general_share:.1f} / {share_first:.1f} / {share_second:.1f}, веса для training_config.json:"
    )
    for name in base_models:
        print(f"    {name}: 1.0")
    print(f"    {specialists[0]}: {share_first * scale:.1f}")
    print(f"    {specialists[1]}: {share_second * scale:.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof_dir", default="oof")
    parser.add_argument("--general", default="general_rubert,rubert,e5small_llm15_len448")
    parser.add_argument("--base", default="general_rubert,e5small_llm15_len448")
    parser.add_argument("--specialists", default="spec_hard_rubert_v2,spec_hard_e5")
    parser.add_argument("--categories", default="Одежда,Обувь,Ювелирные изделия")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--parts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    general, base = args.general.split(","), args.base.split(",")
    specialists, hard = args.specialists.split(","), args.categories.split(",")
    if not set(base) <= set(general):
        raise SystemExit("--base должен быть подмножеством --general")
    if len(specialists) != 2:
        raise SystemExit("--specialists: нужно ровно два специалиста")

    frame = None
    for name in general:
        part = load(args.oof_dir, name)
        frame = part if frame is None else frame.merge(part[["id1", "id2", name]], on=["id1", "id2"])
    for name in specialists:
        frame = frame.merge(load(args.oof_dir, name)[["id1", "id2", name]], on=["id1", "id2"], how="left")
    for name in general + specialists:
        frame[name] = zscore(frame[name].to_numpy())

    cats, target = frame["category"].to_numpy(), frame["target"].to_numpy()
    print("Одиночные модели")
    for name in general:
        print(f"  {macro_pr_auc(cats, target, frame[name].to_numpy())[0]:.5f}  {name}")

    search_general(frame, general, args.top)
    crossfit_specialists(frame, base, specialists, hard, args.seed, args.parts)


if __name__ == "__main__":
    main()
