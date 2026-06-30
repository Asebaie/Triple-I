import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

TRAIN_PATH = "rascar-ai-chem-hack/train.csv"
TEST_PATH = "rascar-ai-chem-hack/test.csv"
OUTPUT = "submission.csv"
N_FOLDS = 10
SEED = 42


def mol_from_smiles(smi):
    try:
        return Chem.MolFromSmiles(str(smi))
    except:
        return None


def morgan_fp(mol, radius=2, nbits=2048):
    if mol is None:
        return np.zeros(nbits)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits), dtype=np.float32)


def maccs_fp(mol):
    if mol is None:
        return np.zeros(167, dtype=np.float32)
    return np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)


def rdkit_fp(mol, nbits=1024):
    if mol is None:
        return np.zeros(nbits, dtype=np.float32)
    return np.array(Chem.RDKFingerprint(mol, fpSize=nbits), dtype=np.float32)


def mol_descriptors(mol):
    if mol is None:
        return np.zeros(22, dtype=np.float32)
    try:
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        return np.array(
            [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                hbd,
                hba,
                hbd + hba,
                hbd * hba,
                Descriptors.TPSA(mol),
                Descriptors.NumRotatableBonds(mol),
                rdMolDescriptors.CalcNumAromaticRings(mol),
                rdMolDescriptors.CalcNumRings(mol),
                Descriptors.FractionCSP3(mol),
                rdMolDescriptors.CalcNumHeavyAtoms(mol),
                rdMolDescriptors.CalcNumHeteroatoms(mol),
                rdMolDescriptors.CalcNumAmideBonds(mol),
                Descriptors.MolMR(mol),
                Descriptors.BertzCT(mol),
                rdMolDescriptors.CalcNumSaturatedRings(mol),
                rdMolDescriptors.CalcNumAliphaticRings(mol),
                Descriptors.HeavyAtomMolWt(mol),
                Descriptors.NumValenceElectrons(mol),
                rdMolDescriptors.CalcNumAromaticCarbocycles(mol),
                rdMolDescriptors.CalcNumAromaticHeterocycles(mol),
            ],
            dtype=np.float32,
        )
    except:
        return np.zeros(22, dtype=np.float32)


def tanimoto(a, b):
    ab = np.dot(a, b)
    denom = np.sum(a) + np.sum(b) - ab
    return float(ab / denom) if denom > 0 else 0.0


def featurize_pair(smi1, smi2):
    mol1 = mol_from_smiles(smi1)
    mol2 = mol_from_smiles(smi2)

    ecfp4_1 = morgan_fp(mol1, 2, 2048)
    ecfp4_2 = morgan_fp(mol2, 2, 2048)
    ecfp6_1 = morgan_fp(mol1, 3, 2048)
    ecfp6_2 = morgan_fp(mol2, 3, 2048)
    maccs1 = maccs_fp(mol1)
    maccs2 = maccs_fp(mol2)
    rdk1 = rdkit_fp(mol1)
    rdk2 = rdkit_fp(mol2)

    d1 = mol_descriptors(mol1)
    d2 = mol_descriptors(mol2)

    sims = np.array(
        [
            tanimoto(ecfp4_1, ecfp4_2),
            tanimoto(ecfp6_1, ecfp6_2),
            tanimoto(maccs1, maccs2),
            tanimoto(rdk1, rdk2),
            d1[2] * d2[3],
            d2[2] * d1[3],
            (d1[2] + d2[2]) * (d1[3] + d2[3]),
            min(d1[0], d2[0]) / max(d1[0], d2[0]) if max(d1[0], d2[0]) > 0 else 0,
            abs(d1[1] - d2[1]),
            d1[6] + d2[6],
            tanimoto(ecfp4_1 * ecfp4_2, ecfp4_1 + ecfp4_2),
        ],
        dtype=np.float32,
    )

    extra = np.concatenate(
        [
            d1 * d2,
            np.abs(d1 - d2) ** 1.5,
        ]
    )

    return np.concatenate(
        [
            ecfp4_1,
            ecfp4_2,
            ecfp4_1 * ecfp4_2,
            np.abs(ecfp4_1 - ecfp4_2),
            ecfp6_1,
            ecfp6_2,
            ecfp6_1 * ecfp6_2,
            np.abs(ecfp6_1 - ecfp6_2),
            maccs1,
            maccs2,
            maccs1 * maccs2,
            rdk1,
            rdk2,
            rdk1 * rdk2,
            d1,
            d2,
            np.abs(d1 - d2),
            d1 + d2,
            sims,
            extra,
        ]
    )


def build_features(df):
    return np.array([featurize_pair(r["SMILES1"], r["SMILES2"]) for _, r in df.iterrows()], dtype=np.float32)


def weighted_accuracy(y_true, y_pred):
    classes, counts = np.unique(y_true, return_counts=True)
    w = {c: 1.0 / cnt for c, cnt in zip(classes, counts)}
    total = sum(w[c] for c in y_true)
    correct = sum(w[y_true[i]] for i in range(len(y_true)) if y_true[i] == y_pred[i])
    return correct / total


def tune_threshold(y_true, probs):
    best_t, best_wa = 0.5, 0
    for t in np.arange(0.05, 0.95, 0.005):
        wa = weighted_accuracy(y_true, (probs > t).astype(int))
        if wa > best_wa:
            best_wa, best_t = wa, t
    return best_t, best_wa


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    train_pairs = {(r["SMILES1"], r["SMILES2"]): r["result"] for _, r in train.iterrows()}

    submission = pd.DataFrame({"id": test["id"], "result": -1})

    known = 0
    for i, row in test.iterrows():
        key = (row["SMILES1"], row["SMILES2"])
        if key in train_pairs:
            submission.at[i, "result"] = train_pairs[key]
            known += 1

    print(f"Exact matches: {known} / {len(test)} ({known/len(test)*100:.1f}%)")

    unknown = test[submission["result"] == -1].copy()
    if len(unknown) == 0:
        submission.to_csv(OUTPUT, index=False)
        return

    X_train_raw = build_features(train)
    X_test_raw = build_features(unknown)

    selector = VarianceThreshold(threshold=0.005)
    X_train = selector.fit_transform(X_train_raw)
    X_test = selector.transform(X_test_raw)

    y_train = train["result"].values
    spw = np.bincount(y_train)[0] / np.bincount(y_train)[1]

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.015,
        "num_leaves": 511,
        "max_depth": -1,
        "min_child_samples": 8,
        "feature_fraction": 0.45,
        "bagging_fraction": 0.75,
        "bagging_freq": 5,
        "scale_pos_weight": spw,
        "reg_alpha": 0.15,
        "reg_lambda": 0.8,
        "min_split_gain": 0.005,
        "verbose": -1,
        "seed": SEED,
        "n_jobs": -1,
    }

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y_train))
    tst = np.zeros(len(unknown))

    for fold, (tr, va) in enumerate(skf.split(X_train, y_train)):
        model = lgb.train(
            params,
            lgb.Dataset(X_train[tr], y_train[tr]),
            num_boost_round=5000,
            valid_sets=[lgb.Dataset(X_train[va], y_train[va])],
            callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(400)],
        )
        oof[va] = model.predict(X_train[va])
        tst += model.predict(X_test) / N_FOLDS

    thresh, wa = tune_threshold(y_train, oof)
    print(f"OOF WA: {wa:.4f} thresh: {thresh:.3f}")

    submission.loc[unknown.index, "result"] = (tst > thresh).astype(int)
    submission.to_csv(OUTPUT, index=False)
    print("saved")


if __name__ == "__main__":
    main()
