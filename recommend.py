import logging
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

MODEL_ARTIFACT_PATH = "artifacts/ae_model_bundle_optuna.pkl"
MOVIE_TITLES_PATH = "archive/movie_titles.csv"
LOGS_DIR = "logs"

os.makedirs(LOGS_DIR, exist_ok=True)

log_file = f"{LOGS_DIR}/recommend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger().info


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


def load_model_artifacts(artifact_path):
    log(f"Загрузка артефактов из {artifact_path}")
    artifact_data = joblib.load(artifact_path)

    model_arch_info = artifact_data["model_architecture_info"]
    model_state_dict = artifact_data["model_state_dict"]
    row_means = artifact_data["row_means"]
    user_map = artifact_data["user_map"]
    movie_map = artifact_data["movie_map"]
    best_val_rmse = artifact_data.get("best_val_rmse", "N/A")

    log(f"Артефакты загружены. RMSE на валидации: {best_val_rmse}")

    model = Autoencoder(num_movies=model_arch_info["num_movies"], hidden_layers=model_arch_info["hidden_layers"])
    model.load_state_dict(model_state_dict)
    model.eval()

    return model, row_means, user_map, movie_map


def load_movie_titles(path):
    log(f"Загрузка названий фильмов из {path}")
    movie_titles_df = pd.DataFrame(columns=["MovieID", "Year", "Title"])
    with open(path, "r", encoding="latin-1") as f:
        for line_num, line in enumerate(f, start=1):
            parts = line.strip().split(",", 2)
            if len(parts) != 3:
                log(f"Предупреждение: Неправильный формат строки {line_num}: '{line.strip()}'")
                continue
            try:
                movie_id = int(parts[0])
                year_str = parts[1]
                year = int(year_str) if year_str != "NULL" else np.nan
                title = parts[2]
                movie_titles_df.loc[len(movie_titles_df)] = [movie_id, year, title]
            except ValueError:
                log(f"Предупреждение: Ошибка преобразования в int в строке {line_num}: '{line.strip()}'")
                continue
    log(f"Загружено {len(movie_titles_df)} названий фильмов")
    return movie_titles_df


def get_recommendations(model, user_id, row_means, user_map, movie_map, movie_titles_df, top_k=10):
    """
    Генерирует топ-K рекомендаций для пользователя.
    """
    if user_id not in user_map:
        log(f"Пользователь {user_id} не найден в обучающем сете.")
        print(f"Ошибка: Пользователь {user_id} не найден в обучающем сете.")
        return []

    user_idx = user_map[user_id]
    device = next(model.parameters()).device

    data_path = "archive"
    user_raw_ratings = {}
    for i in range(1, 5):
        file_path = os.path.join(data_path, f"combined_data_{i}.txt")
        with open(file_path, "r", encoding="latin-1") as f:
            movie_id = None
            for line in f:
                line = line.strip()
                if line.endswith(":"):
                    movie_id = int(line[:-1])
                else:
                    user_id_line, rating, _ = line.split(",")
                    if int(user_id_line) == user_id and movie_id in movie_map:
                        user_raw_ratings[movie_id] = int(rating)

    if not user_raw_ratings:
        log(f"Для пользователя {user_id} не найдено рейтинов в обучающем сете.")
        print(f"Предупреждение: У пользователя {user_id} нет известных рейтинов.")
        return []

    num_movies = len(movie_map)
    user_ratings_vector = np.zeros(num_movies)
    for movie_id, rating in user_raw_ratings.items():
        movie_idx = movie_map[movie_id]
        user_ratings_vector[movie_idx] = rating

    user_mean = row_means[user_idx]
    user_ratings_normalized = user_ratings_vector - user_mean
    user_ratings_tensor = torch.from_numpy(user_ratings_normalized).float().unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_ratings_normalized = model(user_ratings_tensor).squeeze(0).cpu().numpy()

    predicted_ratings_denormalized = predicted_ratings_normalized + user_mean

    rated_movie_indices = [movie_map[mid] for mid in user_raw_ratings.keys() if mid in movie_map]
    predicted_ratings_denormalized[rated_movie_indices] = -np.inf

    top_k_movie_indices = np.argsort(predicted_ratings_denormalized)[::-1][:top_k]
    idx_to_movie = {idx: movie_id for movie_id, idx in movie_map.items()}
    recommended_movie_ids = [idx_to_movie[idx] for idx in top_k_movie_indices if idx in idx_to_movie]

    recommended_movies_df = movie_titles_df[movie_titles_df["MovieID"].isin(recommended_movie_ids)]
    recommended_movies_df = recommended_movies_df.set_index("MovieID").reindex(recommended_movie_ids).reset_index()

    return recommended_movies_df[["MovieID", "Title"]].to_dict("records")


if __name__ == "__main__":
    log("Начало генерации рекомендаций")

    try:
        model, row_means, user_map, movie_map = load_model_artifacts(MODEL_ARTIFACT_PATH)
    except FileNotFoundError:
        log(f"Файл артефактов не найден: {MODEL_ARTIFACT_PATH}")
        print(f"Ошибка: Файл артефактов модели не найден по пути: {MODEL_ARTIFACT_PATH}")
        exit(1)
    except Exception as e:
        log(f"Ошибка загрузки артефактов: {e}")
        print(f"Ошибка загрузки артефактов: {e}")
        exit(1)

    try:
        movie_titles_df = load_movie_titles(MOVIE_TITLES_PATH)
    except FileNotFoundError:
        log(f"Файл с названиями фильмов не найден: {MOVIE_TITLES_PATH}")
        print(f"Ошибка: Файл с названиями фильмов не найден по пути: {MOVIE_TITLES_PATH}")
        exit(1)

    user_ids_to_predict = list(user_map.keys())[:10]

    if not user_ids_to_predict:
        print("Не найдено пользователей для генерации рекомендаций.")
        exit(0)

    all_recommendations = []
    for idx, user_id in enumerate(user_ids_to_predict, 1):
        log(f"Поиск истории просмотров пользователя [{idx}/10]")
        recommendations = get_recommendations(model, user_id, row_means, user_map, movie_map, movie_titles_df, top_k=1)
        if recommendations:
            all_recommendations.append((user_id, recommendations[0]))
        else:
            all_recommendations.append((user_id, {'MovieID': 'N/A', 'Title': 'No recommendations'}))

    print("\n--- Рекомендации ---")
    print(f"User_ID  | Film_ID   | Film_Name")
    for user_id, top_film in all_recommendations:
        print(f"{user_id:<8} | {top_film['MovieID']:<9} | {top_film['Title']}")

    log("Генерация рекомендаций завершена.")
