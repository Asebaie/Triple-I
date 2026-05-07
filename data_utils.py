import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from config import DATA_FILE, GT_FILE, logging


def load_and_preprocess():
    if not os.path.exists(DATA_FILE) or not os.path.exists(GT_FILE):
        raise FileNotFoundError(f"Data files not found.")

    data = np.load(DATA_FILE)
    gt = np.load(GT_FILE)
    h, w, bands = data.shape

    X_all = data.reshape(-1, bands).astype(np.float32)
    y_all = gt.reshape(-1)

    mask_labeled = y_all > 0
    X = X_all[mask_labeled]
    y = y_all[mask_labeled].astype(np.int64)

    classes, counts = np.unique(gt, return_counts=True)
    class_distribution = pd.DataFrame({"class": classes, "pixel_count": counts})
    class_distribution["share"] = class_distribution["pixel_count"] / class_distribution["pixel_count"].sum()

    logging.info(f"Shape: {data.shape}")
    logging.info(f"Classes: {len(classes)}")
    logging.info(f"Pixels: {np.count_nonzero(gt)}")
    logging.info(f"Min class: {class_distribution[class_distribution['class'] != 0]['pixel_count'].min()}")
    logging.info(f"Max class: {class_distribution[class_distribution['class'] != 0]['pixel_count'].max()}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(gt, cmap="jet")
    axes[0].axis("off")

    axes[1].imshow(data[:, :, bands // 2], cmap="gray")
    axes[1].axis("off")

    sample_y, sample_x = np.argwhere(gt > 0)[0]
    axes[2].plot(data[sample_y, sample_x, :])

    plt.tight_layout()
    plt.savefig("results/exploration_plots.png", bbox_inches="tight")
    plt.close()

    return X, y, X_all, y_all, mask_labeled, h, w, bands


def prepare_pytorch_loaders(X_train, y_train, X_test, y_test, batch_size=64):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    train_ds = TensorDataset(torch.tensor(X_train_scaled), torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test_scaled), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    return train_loader, test_loader, scaler