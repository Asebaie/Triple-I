import functools
import logging
import os
from datetime import datetime

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

DATA_PATH = "archive"
ARTIFACTS_DIR = "artifacts"
LOGS_DIR = "logs"
MODEL_ARTIFACT_PATH = os.path.join(ARTIFACTS_DIR, "ae_model_bundle_optuna.pkl")
REPORT_PATH = os.path.join(ARTIFACTS_DIR, "report_optuna.txt")

NUM_USERS_SUBSET = 10000

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

log_file = f"{LOGS_DIR}/netflix_ae_optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger().info


class NetflixDataset(Dataset):
    def __init__(self, data):
        self.data = torch.from_numpy(data).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Autoencoder(nn.Module):
    def __init__(self, num_movies, hidden_layers):
        super().__init__()
        layers = []
        prev_size = num_movies
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        for hidden_size in reversed(hidden_layers[:-1]):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_movies))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_and_preprocess_data(data_path, num_users_subset=NUM_USERS_SUBSET):
    log("Парсинг training set")
    train_frames = []
    for i in range(1, 5):
        file_path = os.path.join(data_path, f"combined_data_{i}.txt")
        log(f"Чтение {file_path}")
        with open(file_path, "r") as f:
            movie_id = None
            for line in f:
                line = line.strip()
                if line.endswith(":"):
                    movie_id = int(line[:-1])
                else:
                    user_id, rating, _ = line.split(",")
                    train_frames.append([int(user_id), movie_id, int(rating)])

    train_df = pd.DataFrame(train_frames, columns=["user_id", "movie_id", "rating"])
    log(f"Загружено {len(train_df)} оценок")

    if num_users_subset:
        log(f"Подвыборка данных: первые {num_users_subset} пользователей")
        top_users = train_df["user_id"].value_counts().head(num_users_subset).index
        train_df = train_df[train_df["user_id"].isin(top_users)]
        log(f"После подвыборки: {len(train_df)} оценок, пользователей: {train_df['user_id'].nunique()}")

    user_map = {uid: i for i, uid in enumerate(train_df["user_id"].unique())}
    movie_map = {mid: j for j, mid in enumerate(train_df["movie_id"].unique())}
    train_df["user_idx"] = train_df["user_id"].map(user_map)
    train_df["movie_idx"] = train_df["movie_id"].map(movie_map)

    num_users = len(user_map)
    num_movies = len(movie_map)
    log(f"Пользователей: {num_users} | Фильмов: {num_movies}")

    log("Создание sparse матрицы")
    sparse_matrix = coo_matrix(
        (train_df["rating"], (train_df["user_idx"], train_df["movie_idx"])),
        shape=(num_users, num_movies),
        dtype=np.float32,
    )
    matrix = sparse_matrix.toarray()

    log("Нормализация (вычитание среднего пользователя)")
    row_mask = matrix > 0
    row_sums = matrix.sum(axis=1)
    row_counts = row_mask.sum(axis=1)
    row_means = np.zeros(num_users)
    row_means[row_counts > 0] = row_sums[row_counts > 0] / row_counts[row_counts > 0]
    matrix_normalized = matrix - row_means[:, None]

    return matrix_normalized, row_means, user_map, movie_map, num_movies


def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    num_examples = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            output = model(batch)
            mask = (batch != 0).float()
            loss = nn.MSELoss()(output * mask, batch * mask)

            num_valid = mask.sum().item()
            if num_valid > 0:
                total_loss += loss.item() * num_valid
                num_examples += num_valid
    model.train()
    if num_examples > 0:
        avg_mse = total_loss / num_examples
        rmse = np.sqrt(avg_mse)
        return rmse
    else:
        return float("inf")


def objective(trial, train_loader, val_loader, num_movies, device):
    hidden_1 = trial.suggest_int("hidden_1", 256, 1024)
    hidden_2 = trial.suggest_int("hidden_2", 128, hidden_1)
    hidden_layers = [hidden_1, hidden_2]

    learning_rate = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    epochs = trial.suggest_int("epochs", 10, 50)

    model = Autoencoder(num_movies, hidden_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            mask = (batch != 0).float()
            loss = criterion(output * mask, batch * mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_rmse = evaluate_model(model, val_loader, device)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse

        # log(f"Trial {trial.number}, Epoch {epoch+1}/{epochs}, Val RMSE: {val_rmse:.6f}")

    return best_val_rmse


if __name__ == "__main__":
    log("Начало процесса обучения автоэнкодера с Optuna")

    log("Загрузка и предварительная обработка данных")
    matrix_normalized, row_means, user_map, movie_map, num_movies = load_and_preprocess_data(
        DATA_PATH, NUM_USERS_SUBSET
    )

    log("Разделение данных на train/val")
    train_matrix, val_matrix = train_test_split(matrix_normalized, test_size=0.1, random_state=42)

    log("Создание DataLoader'ов")
    train_ds = NetflixDataset(train_matrix)
    val_ds = NetflixDataset(val_matrix)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Используется устройство: {device}")

    log("Запуск оптимизации гиперпараметров с Optuna")

    study = optuna.create_study(direction="minimize")

    objective_with_data = functools.partial(
        objective, train_loader=train_loader, val_loader=val_loader, num_movies=num_movies, device=device
    )

    study.optimize(objective_with_data, n_trials=15)

    best_params = study.best_params
    best_val_rmse = study.best_value
    log(f"Лучшие гиперпараметры: {best_params}")
    log(f"Лучший Val RMSE: {best_val_rmse:.6f}")

    log("Обучение финальной модели с лучшими гиперпараметрами на всем train-наборе")

    final_hidden_layers = [best_params["hidden_1"], best_params["hidden_2"]]
    final_model = Autoencoder(num_movies, final_hidden_layers).to(device)
    final_optimizer = optim.Adam(
        final_model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"]
    )
    final_criterion = nn.MSELoss()

    num_epochs = best_params["epochs"]
    best_val_rmse_final = float("inf")

    for epoch in range(num_epochs):
        final_model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            final_optimizer.zero_grad()
            output = final_model(batch)
            mask = (batch != 0).float()
            loss = final_criterion(output * mask, batch * mask)
            loss.backward()
            final_optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        val_rmse = evaluate_model(final_model, val_loader, device)

        if val_rmse < best_val_rmse_final:
            best_val_rmse_final = val_rmse

        log(
            f"Epoch [{epoch + 1}/{num_epochs}], Train MSE: {avg_train_loss:.6f}, Val RMSE: {val_rmse:.6f}, Best Val RMSE: {best_val_rmse_final:.6f}"
        )

    artifact_data = {
        "model_state_dict": final_model.state_dict(),
        "model_architecture_info": {"hidden_layers": final_hidden_layers, "num_movies": num_movies},
        "row_means": row_means,
        "user_map": user_map,
        "movie_map": movie_map,
        "best_val_rmse": best_val_rmse_final,
        "best_params": best_params,
    }
    joblib.dump(artifact_data, MODEL_ARTIFACT_PATH)
    log(f"Финальная модель сохранена в {MODEL_ARTIFACT_PATH}")

    report_content = f"""Отчет по обучению рекомендательной системы (автоэнкодер) с Optuna.

Лучшие гиперпараметры: {best_params}
Финальный Val RMSE (на лучшей эпохе): {best_val_rmse_final:.6f}
"""
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)
    log(f"Отчет сохранен в {REPORT_PATH}")

    log("Обучение завершено.")
