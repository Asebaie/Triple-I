import numpy as np


class NumpyBooster:
    def __init__(self, trees, n_features, sigmoid):
        self.trees = trees
        self.n_features = n_features
        self.sigmoid = sigmoid

    @property
    def num_trees(self):
        return len(self.trees)

    def raw_score(self, matrix):
        total = np.zeros(matrix.shape[0], dtype=np.float64)
        for split_feature, threshold, left_child, right_child, leaf_value in self.trees:
            node = np.zeros(matrix.shape[0], dtype=np.int32)
            while True:
                internal = node >= 0
                if not internal.any():
                    break
                current = node[internal]
                values = matrix[internal, split_feature[current]]
                node[internal] = np.where(values <= threshold[current], left_child[current], right_child[current])
            total += leaf_value[~node]
        return total

    def predict(self, matrix):
        return 1.0 / (1.0 + np.exp(-self.sigmoid * self.raw_score(np.asarray(matrix, dtype=np.float64))))


def _parse_block(lines):
    fields = {}
    for line in lines:
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        fields[key.strip()] = value.strip()
    return fields


def load_model(path):
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()

    blocks = text.split("\n\n")
    header = _parse_block(blocks[0].splitlines())
    n_features = int(header["max_feature_idx"]) + 1
    objective = header.get("objective", "binary sigmoid:1")
    sigmoid = 1.0
    for part in objective.split():
        if part.startswith("sigmoid:"):
            sigmoid = float(part.split(":")[1])
    if not objective.startswith("binary"):
        raise ValueError(f"поддерживается только binary, в файле: {objective}")

    trees = []
    for block in blocks[1:]:
        if not block.lstrip().startswith("Tree="):
            continue
        fields = _parse_block(block.splitlines())
        if int(fields.get("num_cat", "0")) != 0:
            raise ValueError("категориальные сплиты не поддерживаются, обучай без categorical_feature")
        if int(fields.get("is_linear", "0")) != 0:
            raise ValueError("линейные листья не поддерживаются")
        leaf_value = np.fromstring(fields["leaf_value"], sep=" ", dtype=np.float64)
        if int(fields["num_leaves"]) <= 1:
            trees.append(
                (
                    np.zeros(1, dtype=np.int32),
                    np.full(1, np.inf, dtype=np.float64),
                    np.array([-1], dtype=np.int32),
                    np.array([-1], dtype=np.int32),
                    leaf_value,
                )
            )
            continue
        trees.append(
            (
                np.fromstring(fields["split_feature"], sep=" ", dtype=np.int32),
                np.fromstring(fields["threshold"], sep=" ", dtype=np.float64),
                np.fromstring(fields["left_child"], sep=" ", dtype=np.int32),
                np.fromstring(fields["right_child"], sep=" ", dtype=np.int32),
                leaf_value,
            )
        )
    if not trees:
        raise ValueError(f"в файле {path} не найдено деревьев")
    return NumpyBooster(trees, n_features, sigmoid)
