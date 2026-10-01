import numpy as np
from sklearn.metrics import average_precision_score

TEST_PREVALENCE = 0.05


def macro_pr_auc(categories, targets, scores):
    per_category = {}
    for cat in np.unique(categories):
        mask = categories == cat
        y = targets[mask]
        if y.min() == y.max():
            continue
        per_category[int(cat)] = float(average_precision_score(y, scores[mask]))
    if not per_category:
        return 0.0, per_category
    return float(np.mean(list(per_category.values()))), per_category


def macro_pr_auc_at_prevalence(categories, targets, scores, prevalence=TEST_PREVALENCE, repeats=5, seed=0):
    binary = (targets >= 0.5).astype(np.int8)
    values = []
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)
        keep = []
        for cat in np.unique(categories):
            index = np.flatnonzero(categories == cat)
            positive = index[binary[index] == 1]
            negative = index[binary[index] == 0]
            if len(positive) == 0 or len(negative) == 0:
                continue
            wanted = int(round(len(negative) * prevalence / max(1.0 - prevalence, 1e-9)))
            wanted = max(min(wanted, len(positive)), 3)
            keep.append(np.concatenate([rng.choice(positive, wanted, replace=False), negative]))
        if not keep:
            return 0.0, 0.0
        keep = np.concatenate(keep)
        score, _ = macro_pr_auc(categories[keep], targets[keep], scores[keep])
        values.append(score)
    return float(np.mean(values)), float(np.std(values))
