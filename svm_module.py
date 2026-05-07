import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
import optuna
from optuna.samplers import TPESampler
from config import RANDOM_STATE, logging

optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_svm_pipeline(params: dict) -> Pipeline:
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("pca", PCA(
            n_components=params["pca_variance"],
            svd_solver="full",
            random_state=RANDOM_STATE,
        )),
        ("svc", SVC(
            kernel=params["kernel"],
            C=params["C"],
            gamma=params["gamma"],
            class_weight="balanced",
            cache_size=1000,
            random_state=RANDOM_STATE,
        )),
    ])


def plot_pca_justification(X_train):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    pca = PCA(random_state=RANDOM_STATE)
    pca.fit(X_scaled)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    thresholds = [0.90, 0.95, 0.98, 0.99]

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, marker=".")
    for t in thresholds:
        n = int(np.searchsorted(cumulative_variance, t) + 1)
        plt.axhline(t, linestyle="--", label=f"{t:.0%} (n={n})")
        logging.info(f"PCA component for {t:.0%}: {n}")

    plt.grid(True)
    plt.legend()
    plt.savefig("results/pca_variance_plot.png", bbox_inches="tight")
    plt.close()


def get_cv(y_train: np.ndarray) -> StratifiedKFold:
    _, class_counts = np.unique(y_train, return_counts=True)
    min_count = int(class_counts.min())
    n_splits = max(2, min(3, min_count))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def optimize_svm_hyperparameters(X_train: np.ndarray, y_train: np.ndarray, n_trials: int = 12) -> dict:
    cv = get_cv(y_train)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "pca_variance": trial.suggest_categorical("pca_variance", [0.90, 0.95, 0.98, 0.99]),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "linear"]),
            "C": trial.suggest_float("C", 0.1, 100.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1e-1, log=True),
        }
        model = build_svm_pipeline(params)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring="f1_macro", n_jobs=-1,
        )
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params