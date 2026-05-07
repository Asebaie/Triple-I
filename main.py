import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from config import RANDOM_STATE, TARGET_ACCURACY, TARGET_F1_MACRO, device, logging
from data_utils import load_and_preprocess
from svm_module import plot_pca_justification, optimize_svm_hyperparameters, build_svm_pipeline
from cnn_module import SpectralCNN1D, BaselineCNN2D, count_parameters, run_cnn_training, evaluate_cnn


def main():
    X, y, X_all, y_all, mask_labeled, h, w, bands = load_and_preprocess()

    TRAIN_SHARE_BASE = 0.3
    X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
        X, y, train_size=TRAIN_SHARE_BASE, random_state=RANDOM_STATE, shuffle=True, stratify=y
    )

    logging.info(f"Train size: {X_train_base.shape}, Test size: {X_test_base.shape}")

    plot_pca_justification(X_train_base)

    logging.info("Optimizing SVM parameters...")
    best_params_base = optimize_svm_hyperparameters(X_train_base, y_train_base, n_trials=12)
    logging.info(f"Best SVM parameters: {best_params_base}")

    best_model_base = build_svm_pipeline(best_params_base)
    best_model_base.fit(X_train_base, y_train_base)

    y_pred_base = best_model_base.predict(X_test_base)
    acc_base = (y_pred_base == y_test_base).mean()
    logging.info(f"SVM baseline accuracy: {acc_base:.4f}")

    labels = np.unique(y)
    cm = confusion_matrix(y_test_base, y_pred_base, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    plt.savefig("results/svm_confusion_matrix.png", bbox_inches="tight")
    plt.close()

    pred_all_base = np.zeros_like(y_all, dtype=np.uint8)
    pred_all_base[mask_labeled] = best_model_base.predict(X_all[mask_labeled]).astype(np.uint8)
    pred_map_base = pred_all_base.reshape(h, w)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(y_all.reshape(h, w), cmap="jet", vmin=0, vmax=int(y.max()))
    axes[0].axis("off")
    axes[1].imshow(pred_map_base, cmap="jet", vmin=0, vmax=int(y.max()))
    axes[1].axis("off")
    plt.savefig("results/svm_map_base.png", bbox_inches="tight")
    plt.close()

    TRAIN_SHARES = np.round(np.arange(0.1, 1.0, 0.1), 1)
    svm_results = []

    for train_share in TRAIN_SHARES:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, train_size=float(train_share), random_state=RANDOM_STATE, shuffle=True, stratify=y
        )
        best_p = optimize_svm_hyperparameters(X_tr, y_tr, n_trials=8)
        model = build_svm_pipeline(best_p)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(y_te, y_pred)
        f1_mac = f1_score(y_te, y_pred, average="macro", zero_division=0)
        f1_wt = f1_score(y_te, y_pred, average="weighted", zero_division=0)

        svm_results.append({
            "train_share": train_share,
            "accuracy": acc,
            "f1_macro": f1_mac,
            "f1_weighted": f1_wt
        })
        logging.info(f"SVM share {train_share:.1f}: acc={acc:.4f}, f1_macro={f1_mac:.4f}")

    svm_res_df = pd.DataFrame(svm_results)
    svm_res_df.to_csv("results/svm_shares_metrics.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(svm_res_df["train_share"], svm_res_df["accuracy"], marker="o", label="Accuracy")
    plt.plot(svm_res_df["train_share"], svm_res_df["f1_macro"], marker="o", label="F1 macro")
    plt.plot(svm_res_df["train_share"], svm_res_df["f1_weighted"], marker="o", label="F1 weighted")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/svm_vs_shares.png", bbox_inches="tight")
    plt.close()

    logging.info("Deep learning experiments...")
    y_cnn = y - 1
    num_classes = len(np.unique(y_cnn))

    cnn_results = []
    best_cnn_model = None
    best_cnn_scaler = None
    best_cnn_share = 1.0
    found_sufficient_volume = False
    sufficient_share = None

    for train_share in TRAIN_SHARES:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_cnn, train_size=float(train_share), random_state=RANDOM_STATE, shuffle=True, stratify=y_cnn
        )

        model, history, scaler = run_cnn_training(
            X_tr, y_tr, X_te, y_te, num_classes, bands, epochs=40, batch_size=64
        )

        val_loader = DataLoader(
            TensorDataset(
                torch.tensor(scaler.transform(X_te).astype(np.float32)),
                torch.tensor(y_te, dtype=torch.long)
            ),
            batch_size=512,
            shuffle=False
        )

        acc, f1_mac, f1_wt, _ = evaluate_cnn(model, val_loader)

        cnn_results.append({
            "train_share": train_share,
            "accuracy": acc,
            "f1_macro": f1_mac,
            "f1_weighted": f1_wt
        })

        logging.info(f"1D CNN share {train_share:.1f}: acc={acc:.4f}, f1_macro={f1_mac:.4f}")

        if not found_sufficient_volume and acc >= TARGET_ACCURACY and f1_mac >= TARGET_F1_MACRO:
            sufficient_share = train_share
            found_sufficient_volume = True
            logging.info(f"Sufficient training share found: {sufficient_share}")

        if best_cnn_model is None or f1_mac > max([r["f1_macro"] for r in cnn_results[:-1]] + [0]):
            best_cnn_model = model
            best_cnn_scaler = scaler
            best_cnn_share = train_share

    cnn_res_df = pd.DataFrame(cnn_results)
    cnn_res_df.to_csv("results/cnn_shares_metrics.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(cnn_res_df["train_share"], cnn_res_df["accuracy"], marker="o", label="Accuracy")
    plt.plot(cnn_res_df["train_share"], cnn_res_df["f1_macro"], marker="o", label="F1 macro")
    plt.plot(cnn_res_df["train_share"], cnn_res_df["f1_weighted"], marker="o", label="F1 weighted")
    plt.grid(True)
    plt.legend()
    plt.savefig("results/cnn_vs_shares.png", bbox_inches="tight")
    plt.close()

    logging.info(f"Target accuracy: {TARGET_ACCURACY}, Target F1 macro: {TARGET_F1_MACRO}")
    if found_sufficient_volume:
        logging.info(f"Sufficient training volume to match or beat 2D CNN model: {sufficient_share * 100:.1f}%")
    else:
        logging.info("Could not match the exact 2D baseline targets. Closest run:")
        logging.info(cnn_res_df.loc[cnn_res_df["f1_macro"].idxmax()].to_dict())

    model_1d = SpectralCNN1D(spectral_length=bands, num_classes=num_classes)
    model_2d = BaselineCNN2D(in_channels=bands, num_classes=num_classes, patch_size=9)

    p_1d = count_parameters(model_1d)
    p_2d = count_parameters(model_2d)

    logging.info(f"Parameters 1D CNN: {p_1d:,}")
    logging.info(f"Parameters 2D CNN (Baseline): {p_2d:,}")
    logging.info(f"Parameter reduction gain: {p_2d / p_1d:.2f}x")

    X_all_sc = torch.tensor(best_cnn_scaler.transform(X_all).astype(np.float32))
    pred_all_cnn = np.zeros(len(y_all), dtype=np.uint8)
    best_cnn_model.eval()
    with torch.no_grad():
        for start in range(0, len(X_all_sc), 4096):
            batch = X_all_sc[start:start + 4096].to(device)
            pred_all_cnn[start:start + 4096] = best_cnn_model(batch).argmax(1).cpu().numpy()

    pred_map_cnn = np.zeros_like(y_all, dtype=np.uint8)
    pred_map_cnn[mask_labeled] = (pred_all_cnn[mask_labeled] + 1).astype(np.uint8)
    pred_map_map = pred_map_cnn.reshape(h, w)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(y_all.reshape(h, w), cmap="jet", vmin=0, vmax=16)
    axes[0].axis("off")
    axes[1].imshow(pred_map_map, cmap="jet", vmin=0, vmax=16)
    axes[1].axis("off")
    plt.savefig("results/cnn_best_map.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()