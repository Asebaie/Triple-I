import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
import logging
import os
from datetime import datetime

os.makedirs('logs', exist_ok=True, mode=0o755)
log_file = f"logs/best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger().info

log("ЗАПУСК: УЛУЧШЕНИЕ ЛУЧШЕГО КОДА → WMAE < 45 000")




log("Загрузка данных...")
try:
    train = pd.read_csv('datasets/hackathon_income_train.csv', sep=';', decimal=',', encoding='utf-8', low_memory=False)
    test = pd.read_csv('datasets/hackathon_income_test.csv', sep=';', decimal=',', encoding='utf-8', low_memory=False)
    log(f"Train: {train.shape} | Test: {test.shape}")
except Exception as e:
    log(f"ОШИБКА ЗАГРУЗКИ: {e}")
    raise

y_raw = train['target'].values
weights = train['w'].values
id_test = test['id'].copy()
y = np.log1p(y_raw)

X = train.drop(columns=['id', 'target', 'w'])
X_test = test.drop(columns=['id'])




log("Target Encoding (CV-safe)...")
cat_features = X.select_dtypes('object').columns.tolist()
log(f"Категориальных фич: {len(cat_features)}")


def cv_safe_encode(train_df, test_df, col, target_col='target', folds=5):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    encoded_train = np.zeros(len(train_df))
    global_mean = train_df[target_col].mean()

    for tr_idx, val_idx in kf.split(train_df):
        tr = train_df.iloc[tr_idx]
        mean_map = tr.groupby(col)[target_col].mean()
        encoded_train[val_idx] = train_df.iloc[val_idx][col].map(mean_map).fillna(global_mean)

    test_encoded = test_df[col].map(train_df.groupby(col)[target_col].mean()).fillna(global_mean)
    return encoded_train, test_encoded


for col in cat_features:
    try:
        X[col + '_enc'], X_test[col + '_enc'] = cv_safe_encode(
            train.assign(target=y_raw), test, col
        )
        log(f"  {col} → {col}_enc")
    except Exception as e:
        log(f"  ОШИБКА в {col}: {e}")




log("Добавление частоты для частых категорий...")
for col in cat_features:
    freq = X[col].value_counts()
    top_cats = freq[freq > 100].index
    if len(top_cats) > 0:
        X[col + '_freq'] = X[col].isin(top_cats).astype(int)
        X_test[col + '_freq'] = X_test[col].isin(top_cats).astype(int)
        log(f"  {col} → {col}_freq (частых: {len(top_cats)})")




log("Импутация...")
num_cols = X.select_dtypes('number').columns.tolist()
log(f"Числовых колонок: {len(num_cols)}")

if len(num_cols) > 0:
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    X_test[num_cols] = X_test[num_cols].fillna(X[num_cols].median())

for col in cat_features:
    X[col] = X[col].astype(str).fillna('missing')
    X_test[col] = X_test[col].astype(str).fillna('missing')

log("Импутация завершена")




log("Обучение CatBoost...")
model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.04,
    depth=7,
    l2_leaf_reg=8,
    loss_function='RMSE',
    random_seed=42,
    verbose=200,
    early_stopping_rounds=150,
    cat_features=cat_features
)




log("CV-валидация (5 фолдов)...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (tr, val) in enumerate(kf.split(X)):
    log(f"--- ФОЛД {fold + 1}/5 ---")
    pool_tr = Pool(X.iloc[tr], y[tr], cat_features=cat_features)
    pool_val = Pool(X.iloc[val], y[val], cat_features=cat_features)

    model.fit(pool_tr, eval_set=pool_val)
    pred = np.expm1(model.predict(X.iloc[val]))
    wmae = np.mean(weights[val] * np.abs(y_raw[val] - pred))
    cv_scores.append(wmae)
    log(f"WMAE: {wmae:,.0f}")

log(f"CV WMAE: {np.mean(cv_scores):,.0f} ± {np.std(cv_scores):,.0f}")




log("Финальное обучение на всех данных...")
pool_full = Pool(X, y, cat_features=cat_features)
model.fit(pool_full)

pred_final = np.expm1(model.predict(X_test))




submission = pd.DataFrame({'id': id_test, 'target': pred_final})
submission_path = 'sample_submission.csv'
submission.to_csv(submission_path, index=False)
log(f"САБМИТ СОХРАНЁН: {submission_path}")
log(f"Mean target: {pred_final.mean():,.0f} | Std: {pred_final.std():,.0f}")




report = f"""
=== ОТЧЁТ ===
WMAE (CV): {np.mean(cv_scores):,.0f} ± {np.std(cv_scores):,.0f}
Mean target: {pred_final.mean():,.0f}
Features: {X.shape[1]}
Модель: CatBoost (depth=7, iter=2000)
Target Encoding: CV-safe
Частота: для категорий > 100
"""
with open('report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
log("report.txt — СОЗДАН")

log("ВСЁ ЗАВЕРШЕНО УСПЕШНО")
