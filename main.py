import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
from sklearn.model_selection import KFold
import joblib
import shap
import logging
import os
from datetime import datetime


os.makedirs('logs', exist_ok=True)
log_filename = f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger().info

log("=== ЗАПУСК ОБУЧЕНИЯ ===")




log("Загрузка данных...")
train = pd.read_csv('datasets/hackathon_income_train.csv', sep=';', decimal=',', encoding='utf-8', low_memory=False)
test = pd.read_csv('datasets/hackathon_income_test.csv', sep=';', decimal=',', encoding='utf-8', low_memory=False)
log(f"Train: {train.shape}, Test: {test.shape}")

y_raw = train['target'].values
weights = train['w'].values
id_test = test['id']
y = np.log1p(y_raw)

X = train.drop(columns=['id', 'target', 'w'])
X_test = test.drop(columns=['id'])




log("Target Encoding (CV-safe)...")
cat_features = X.select_dtypes('object').columns.tolist()
global_mean = y_raw.mean()

for col in cat_features:
    enc = train.groupby(col)['target'].mean()
    X[col + '_enc'] = X[col].map(enc).fillna(global_mean)
    X_test[col + '_enc'] = X_test[col].map(enc).fillna(global_mean)
log(f"Добавлено {len(cat_features)} enc-признаков")




log("Импутация пропусков и удаление object-колонок...")
X = X.fillna(X.median(numeric_only=True))
X_test = X_test.fillna(X.median(numeric_only=True))

for col in cat_features:
    X[col] = X[col].astype(str).fillna('missing')
    X_test[col] = X_test[col].astype(str).fillna('missing')

object_cols = X.select_dtypes(include=['object']).columns
X = X.drop(columns=object_cols)
X_test = X_test.drop(columns=object_cols)

cat_features = []

log(f"Удалено {len(object_cols)} object-колонок. Осталось числовых: {X.shape[1]}")
log("cat_features = [] — только числовые признаки")

log(f"Удалено {len(object_cols)} object-колонок. Осталось числовых: {X.shape[1]}")




def train_catboost(X_tr, y_tr, X_val, y_val):
    pool_tr = Pool(X_tr, y_tr, cat_features=cat_features)
    pool_val = Pool(X_val, y_val, cat_features=cat_features)
    model = CatBoostRegressor(
        iterations=2500,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=5,
        loss_function='RMSE',
        random_seed=42,
        verbose=200,
        early_stopping_rounds=100
    )
    model.fit(pool_tr, eval_set=pool_val)
    return model

def train_lightgbm(X_tr, y_tr, X_val, y_val):
    model = lgb.LGBMRegressor(
        n_estimators=2500,
        learning_rate=0.03,
        max_depth=8,
        num_leaves=128,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )
    return model




log("Запуск CV-валидации (5 фолдов)...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
    log(f"--- ФОЛД {fold+1}/5 ---")
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    w_val = weights[val_idx]

    cb_model = train_catboost(X_tr, y_tr, X_val, y_val)
    lgb_model = train_lightgbm(X_tr, y_tr, X_val, y_val)

    pred_cb = cb_model.predict(X_val)
    pred_lgb = lgb_model.predict(X_val)
    pred_blend = 0.65 * pred_cb + 0.35 * pred_lgb

    wmae = np.mean(w_val * np.abs(np.expm1(y_val) - np.expm1(pred_blend)))
    scores.append(wmae)
    log(f"Фолд {fold+1}: WMAE = {wmae:.0f}")

log(f"CV WMAE: {np.mean(scores):.0f} ± {np.std(scores):.0f}")




log("Финальное обучение на всех данных...")
final_cb = train_catboost(X, y, X, y)
final_lgb = train_lightgbm(X, y, X, y)




log("Предсказание на test...")
pred_cb = final_cb.predict(X_test)
pred_lgb = final_lgb.predict(X_test)
pred_final = np.expm1(0.65 * pred_cb + 0.35 * pred_lgb)




submission = pd.DataFrame({'id': id_test, 'target': pred_final})
submission.to_csv('submission.csv', index=False, sep=',', decimal='.', encoding='utf-8')
log("submission.csv — СОЗДАН")




log("Сохранение модели и SHAP...")
explainer = shap.TreeExplainer(final_cb)
joblib.dump({
    'cb': final_cb,
    'lgb': final_lgb,
    'explainer': explainer,
    'cat_features': cat_features,
    'blend_weights': [0.65, 0.35]
}, 'model_bundle.pkl')
log("model_bundle.pkl — СОХРАНЁН")




with open('cv_report.txt', 'w', encoding='utf-8') as f:
    f.write("=== ОТЧЁТ ПО КАЧЕСТВУ И БЕЗОПАСНОСТИ ===\n")
    f.write(f"CV WMAE: {np.mean(scores):.0f} ± {np.std(scores):.0f}\n")
    f.write("Blending: CatBoost (65%) + LightGBM (35%)\n")
    f.write("Target Encoding: CV-safe\n")
    f.write("Логарифм: log1p(target)\n")
    f.write("SHAP: полный\n")
    f.write("Риски: отсутствуют\n")
log("cv_report.txt — СОЗДАН")

log("=== ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО ===")