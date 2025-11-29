import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings('ignore')

print("Загрузка...")
train = pd.read_csv('datasets/hackathon_income_train.csv', sep=';', decimal=',', encoding='utf-8', low_memory=False)
test = pd.read_csv('datasets/hackathon_income_test.csv', sep=';', decimal=',', encoding='utf-8', low_memory=False)

y_raw = train['target'].values
weights = train['w'].values
id_test = test['id']




y = np.log1p(y_raw)

X = train.drop(columns=['id', 'target', 'w'])
X_test = test.drop(columns=['id'])




print("Target Encoding (без утечек)...")
cat_features = X.select_dtypes('object').columns.tolist()


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
    X[col + '_enc'], X_test[col + '_enc'] = cv_safe_encode(
        train.assign(target=y_raw), test, col
    )




print("Импутация...")
num_cols = X.select_dtypes('number').columns
X[num_cols] = X[num_cols].fillna(X[num_cols].median())
X_test[num_cols] = X_test[num_cols].fillna(X[num_cols].median())

for col in cat_features:
    X[col] = X[col].astype(str).fillna('missing')
    X_test[col] = X_test[col].astype(str).fillna('missing')




print("Обучение...")
model = CatBoostRegressor(
    iterations=1500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=10,
    loss_function='RMSE',
    random_seed=42,
    verbose=200,
    early_stopping_rounds=100,
    cat_features=cat_features
)




kf = KFold(5, shuffle=True, random_state=42)
cv_scores = []

for tr, val in kf.split(X):
    pool_tr = Pool(X.iloc[tr], y[tr], cat_features=cat_features)
    pool_val = Pool(X.iloc[val], y[val], cat_features=cat_features)

    model.fit(pool_tr, eval_set=pool_val)
    pred_log = model.predict(X.iloc[val])
    pred = np.expm1(pred_log)
    wmae = np.mean(weights[val] * np.abs(y_raw[val] - pred))
    cv_scores.append(wmae)

print(f"CV WMAE: {np.mean(cv_scores):.0f} ± {np.std(cv_scores):.0f}")




pool_full = Pool(X, y, cat_features=cat_features)
model.fit(pool_full)

pred_final = np.expm1(model.predict(X_test))




submission = pd.DataFrame({'id': id_test, 'target': pred_final})
submission.to_csv('sample_submission.csv', index=False)
print("sample_submission.csv — ГОТОВ")
print(f"Mean target: {pred_final.mean():.0f}")
