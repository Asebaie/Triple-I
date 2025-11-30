import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
import logging
import os
from datetime import datetime

os.makedirs('logs', exist_ok=True)
log_file = f"logs/best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
)
log = logging.getLogger().info

log("ФИНАЛЬНЫЙ УДАР: ТВОЙ КОД +ОБЩЕНИЕ + 3 УЛУЧШЕНИЯ → WMAE < 44 000")




log("Загрузка...")
train = pd.read_csv('datasets/hackathon_income_train.csv', sep=';', decimal=',', encoding='utf-8', low_memory=False)
test = pd.read_csv('datasets/hackathon_income_test.csv', sep=';', decimal=',', encoding='utf-8', low_memory=False)
log(f"Train: {train.shape} | Test: {test.shape}")

y_raw = train['target'].values
weights = train['w'].values
id_test = test['id'].copy()

X = train.drop(columns=['id', 'target', 'w'])
X_test = test.drop(columns=['id'])




log("Подготовка фич...")
cat_features = X.select_dtypes('object').columns.tolist()
log(f"Категориальных: {len(cat_features)}")

X[cat_features] = X[cat_features].fillna('missing').astype(str)
X_test[cat_features] = X_test[cat_features].fillna('missing').astype(str)

X_lgb = X.copy()
X_test_lgb = X_test.copy()
X_lgb[cat_features] = X_lgb[cat_features].astype('category')
X_test_lgb[cat_features] = X_test_lgb[cat_features].astype('category')




log("Импутация...")
num_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
X[num_cols] = X[num_cols].fillna(X[num_cols].median())
X_test[num_cols] = X_test[num_cols].fillna(X[num_cols].median())
X_lgb[num_cols] = X_lgb[num_cols].fillna(X_lgb[num_cols].median())
X_test_lgb[num_cols] = X_test_lgb[num_cols].fillna(X_test[num_cols].median())




log("CV + Оптимизация весов...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
pred_cat = np.zeros(len(X_test))
pred_lgb = np.zeros(len(X_test))
cv_scores = []
best_w = 0.7

for fold, (tr, val) in enumerate(kf.split(X)):
    log(f"--- ФОЛД {fold+1}/5 ---")
    X_tr_cat, X_val_cat = X.iloc[tr], X.iloc[val]
    X_tr_lgb, X_val_lgb = X_lgb.iloc[tr], X_lgb.iloc[val]
    y_tr, y_val = y_raw[tr], y_raw[val]
    w_val = weights[val]

    model_cat = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=10,
        loss_function='RMSE',
        random_seed=42,
        verbose=0,
        early_stopping_rounds=100,
        cat_features=cat_features
    )
    model_cat.fit(X_tr_cat, y_tr, eval_set=(X_val_cat, y_val))
    p_cat = model_cat.predict(X_val_cat)
    p_test_cat = model_cat.predict(X_test)

    model_lgb = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    model_lgb.fit(X_tr_lgb, y_tr, eval_set=[(X_val_lgb, y_val)],
                  eval_metric='rmse', callbacks=[lambda x: True])

    p_lgb = model_lgb.predict(X_val_lgb)
    p_test_lgb = model_lgb.predict(X_test_lgb)

    best_wmae = float('inf')
    for w in np.arange(0.5, 0.91, 0.05):
        pred = w * p_cat + (1 - w) * p_lgb
        wmae = np.mean(w_val * np.abs(y_val - pred))
        if wmae < best_wmae:
            best_wmae = wmae
            best_w = w
    log(f"Лучший вес: {best_w:.2f} → WMAE: {best_wmae:,.0f}")

    pred_val = best_w * p_cat + (1 - best_w) * p_lgb
    wmae = np.mean(w_val * np.abs(y_val - pred_val))
    cv_scores.append(wmae)

    pred_cat += p_test_cat / 5
    pred_lgb += p_test_lgb / 5

log(f"CV WMAE: {np.mean(cv_scores):,.0f} ± {np.std(cv_scores):,.0f}")




pred_final = best_w * pred_cat + (1 - best_w) * pred_lgb
train_mean = y_raw.mean()
pred_final = pred_final * (train_mean / pred_final.mean())
log(f"Mean target (after fix): {pred_final.mean():,.0f}")




submission = pd.DataFrame({'id': id_test, 'target': pred_final})
submission.to_csv('sample_submission.csv', index=False)
log("САБМИТ: sample_submission.csv — ГОТОВ")




with open('report.txt', 'w') as f:
    f.write(f"WMAE (CV): {np.mean(cv_scores):,.0f}\n")
    f.write(f"Blend weight: {best_w:.2f}\n")
    f.write(f"Mean target: {pred_final.mean():,.0f}\n")
log("report.txt — готов")
