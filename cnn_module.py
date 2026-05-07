import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from config import device
from data_utils import prepare_pytorch_loaders

class SpectralCNN1D(nn.Module):
    def __init__(self, spectral_length: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x.unsqueeze(1)))

class BaselineCNN2D(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, patch_size: int = 9):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (out.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate_cnn(model, loader):
    model.eval()
    preds_all, true_all = [], []
    for X_batch, y_batch in loader:
        out = model(X_batch.to(device)).argmax(1).cpu()
        preds_all.append(out)
        true_all.append(y_batch)
    preds = torch.cat(preds_all).numpy()
    true = torch.cat(true_all).numpy()
    return (
        accuracy_score(true, preds),
        f1_score(true, preds, average="macro", zero_division=0),
        f1_score(true, preds, average="weighted", zero_division=0),
        preds,
    )

def run_cnn_training(X_tr_np, y_tr_np, X_te_np, y_te_np, num_classes, bands, epochs=40, batch_size=64, lr=1e-3):
    class_counts = np.bincount(y_tr_np)
    weights = class_counts.sum() / (len(class_counts) * (class_counts + 1e-9))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32).to(device)
    )

    train_loader, test_loader, scaler = prepare_pytorch_loaders(
        X_tr_np, y_tr_np, X_te_np, y_te_np, batch_size
    )

    model = SpectralCNN1D(spectral_length=bands, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_acc, val_f1_mac, val_f1_wt, _ = evaluate_cnn(model, test_loader)
        scheduler.step()
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_acc": val_acc,
            "val_f1_mac": val_f1_mac,
            "val_f1_wt": val_f1_wt,
        })

    return model, pd.DataFrame(history), scaler