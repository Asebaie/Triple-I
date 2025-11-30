from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Income Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_cat = None
model_lgb = None
explainer = None
cat_features = []
blend_weight = 0.5
feature_names = []


class ClientData(BaseModel):
    data: Dict[str, Any]


class PredictionResponse(BaseModel):
    predicted_income: float
    shap_explanation: Optional[Dict[str, Any]] = None


@app.on_event('startup')
def startup_event():
    global model_cat, model_lgb, explainer, cat_features, blend_weight, feature_names
    logger.info("Загрузка модели и метаданных препроцессинга...")
    try:
        bundle_path = 'model_bundle.pkl'
        bundle = joblib.load(bundle_path)

        model_cat = bundle['catboost']
        model_lgb = bundle['lightgbm']
        explainer = bundle['explainer']
        cat_features = bundle['cat_features']
        blend_weight = bundle['blend_weight']
        feature_names = bundle['feature_names']

        logger.info("Модель и метаданные успешно загружены.")
        logger.info(f"Количество признаков: {len(feature_names)}")
        logger.info(f"Категориальных признаков: {len(cat_features)}")
        logger.info(f"Blend weight: CatBoost={blend_weight:.2f}, LightGBM={1 - blend_weight:.2f}")

    except FileNotFoundError:
        logger.error("Файл model_bundle.pkl не найден.")
        raise RuntimeError("Файл model_bundle.pkl не найден.")
    except KeyError as e:
        logger.error(f"Ключ {e} не найден в model_bundle.pkl.")
        raise RuntimeError(f"model_bundle.pkl повреждён или устарел: {e}")
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        raise RuntimeError(f"Ошибка при загрузке модели: {e}")


def safe_float(value):
    if value is None or value == '' or value == 'null' or value == 'nan':
        return np.nan
    if isinstance(value, str):
        value = value.replace(',', '.')
    try:
        return float(value)
    except:
        return np.nan


@app.post("/predict", response_model=PredictionResponse)
async def predict_income(client_data: ClientData):
    try:
        logger.info("Получен запрос на предсказание.")

        input_dict = client_data.data
        df_input = pd.DataFrame([input_dict])

        df_input = df_input.drop(columns=['id', 'target', 'w', 'dt'], errors='ignore')

        for col in df_input.columns:
            if col in cat_features:
                df_input[col] = df_input[col].fillna('missing').astype(str)
            else:
                df_input[col] = df_input[col].apply(safe_float)

        num_cols = [col for col in df_input.columns if col not in cat_features]
        medians = df_input[num_cols].median()
        df_input[num_cols] = df_input[num_cols].fillna(medians)

        missing_cols = {}
        for col in feature_names:
            if col not in df_input.columns:
                if col in cat_features:
                    missing_cols[col] = 'missing'
                else:
                    missing_cols[col] = 0.0

        if missing_cols:
            df_missing = pd.DataFrame([missing_cols])
            df_input = pd.concat([df_input, df_missing], axis=1)

        df_input = df_input[feature_names]

        for col in cat_features:
            if col in df_input.columns:
                df_input[col] = df_input[col].astype(str)

        logger.debug(f"Входные данные после препроцессинга: {df_input.shape}")

        pred_cat = model_cat.predict(df_input)

        df_input_lgb = df_input.copy()
        for col in cat_features:
            if col in df_input_lgb.columns:
                df_input_lgb[col] = df_input_lgb[col].astype('category')

        pred_lgb = model_lgb.predict(df_input_lgb)

        pred_blend = blend_weight * pred_cat + (1 - blend_weight) * pred_lgb
        predicted_income = float(pred_blend[0])

        shap_values = None
        try:
            shap_values_raw = explainer.shap_values(df_input.iloc[[0]])
            shap_values = {
                "base_value": float(explainer.expected_value),
                "shap_values": shap_values_raw[0].tolist(),
                "features": df_input.columns.tolist(),
                "feature_values": df_input.iloc[0].tolist()
            }
            logger.debug("SHAP explanation сгенерировано.")
        except Exception as e_shap:
            logger.warning(f"Не удалось получить SHAP explanation: {e_shap}")

        logger.info(f"Предсказан доход: {predicted_income:.2f}")

        return PredictionResponse(
            predicted_income=predicted_income,
            shap_explanation=shap_values
        )

    except Exception as e:
        logger.error(f"Ошибка в /predict: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке запроса: {e}")


@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model_cat is not None and model_lgb is not None}
