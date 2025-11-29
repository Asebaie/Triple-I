'''
--- Запуск ---
uvicorn app:app --reload --port 8000
API будет доступно по http://127.0.0.1:8000
Документация по http://127.0.0.1:8000/docs
'''

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Income Prediction API", version="1.0.0")

model_bundle = None
target_encoders = {}
global_mean = 0.0
median_values = pd.Series()
feature_columns = []

class ClientData(BaseModel):
    data: Dict[str, Any]

class PredictionResponse(BaseModel):
    predicted_income: float
    shap_explanation: Optional[Dict[str, Any]] = None

@app.on_event('startup')
def startup_event():
    global model_bundle, target_encoders, global_mean, median_values, feature_columns
    logger.info("Загрузка модели и метаданных препроцессинга...")
    try:
        bundle_path = '../MLDS/model_bundle.pkl'
        bundle = joblib.load(bundle_path)

        model_bundle = bundle
        target_encoders = bundle['target_encoders']
        global_mean = bundle['global_mean']
        median_values = bundle['median_values']
        feature_columns = bundle['feature_columns']

        logger.info("Модель и метаданные успешно загружены.")
        logger.info(f"Количество финальных признаков: {len(feature_columns)}")
        logger.info(f"Количество категориальных признаков (для Target Encoding): {len(target_encoders)}")

    except FileNotFoundError:
        logger.error("Файл model_bundle.pkl не найден.")
        raise RuntimeError("Файл model_bundle.pkl не найден.")
    except KeyError as e:
        logger.error(f"Ключ {e} не найден в model_bundle.pkl. Убедитесь, что main.py был обновлён и model_bundle.pkl пересоздан.")
        raise RuntimeError(f"model_bundle.pkl повреждён или устарел: {e}")
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        raise RuntimeError(f"Ошибка при загрузке модели: {e}")

@app.post("/predict", response_model=PredictionResponse)
async def predict_income(client_data: ClientData):
    try:
        logger.info("Получен запрос на предсказание.")

        input_dict = client_data.data
        df_input = pd.DataFrame([input_dict])

        df_input = df_input.drop(columns=['id', 'target', 'w'], errors='ignore')

        for col, enc_map in target_encoders.items():
            new_col_name = col + '_enc'
            df_input[new_col_name] = df_input[col].map(enc_map).fillna(global_mean)

        df_input = df_input.fillna(median_values)

        for col in target_encoders.keys():
            df_input[col] = df_input[col].astype(str).fillna('missing')

        df_input = df_input.drop(columns=list(target_encoders.keys()))

        for col in feature_columns:
            if col not in df_input.columns:
                df_input[col] = median_values.get(col, 0)
        df_input = df_input[feature_columns]

        logger.debug(f"Входные данные после препроцессинга: {df_input.shape}")

        cb_model = model_bundle['cb']
        lgb_model = model_bundle['lgb']
        blend_weights = model_bundle['blend_weights']
        explainer = model_bundle['explainer']

        pred_cb = cb_model.predict(df_input)
        pred_lgb = lgb_model.predict(df_input)

        pred_blend_log = blend_weights[0] * pred_cb + blend_weights[1] * pred_lgb

        predicted_income_log = pred_blend_log[0]
        predicted_income = np.expm1(predicted_income_log)

        shap_values = None
        try:
            shap_values_raw = explainer.shap_values(df_input.iloc[[0]])
            shap_values = {
                "base_value": float(explainer.expected_value),
                "shap_values": shap_values_raw[0].tolist(),
                "features": df_input.columns.tolist(),
                "feature_values": df_input.iloc[0].tolist()
            }
            logger.debug(f"SHAP explanation сгенерировано.")
        except Exception as e_shap:
            logger.warning(f"Не удалось получить SHAP explanation: {e_shap}")

        logger.info(f"Предсказан доход: {predicted_income:.2f}")

        return PredictionResponse(
            predicted_income=predicted_income,
            shap_explanation=shap_values
        )

    except Exception as e:
        logger.error(f"Ошибка в /predict: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке запроса: {e}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model_bundle is not None}
