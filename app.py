import joblib 
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.preprocessing import StandardScaler
import os

app = FastAPI(
    title="Spotify Track Popularity Prediction API",
    description="API для предсказания популярности треков на Spotify с использованием линейной регрессии",
    version="1.0.0"
)

# Загрузка модели и скейлера
try:
    # Предполагаем, что модель и скейлер сохранены в файлы
    model = joblib.load('linear_model.pkl')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    raise RuntimeError(f"Ошибка загрузки модели: {str(e)}")

class TrackFeatures(BaseModel):
    danceability: float = Field(..., example=0.8, ge=0.0, le=1.0)
    energy: float = Field(..., example=0.7, ge=0.0, le=1.0)
    key: int = Field(..., example=5, ge=-1, le=11)
    loudness: float = Field(..., example=-5.2)
    mode: int = Field(..., example=1, ge=0, le=1)
    speechiness: float = Field(..., example=0.05, ge=0.0, le=1.0)
    acousticness: float = Field(..., example=0.2, ge=0.0, le=1.0)
    instrumentalness: float = Field(..., example=0.1, ge=0.0, le=1.0)
    liveness: float = Field(..., example=0.1, ge=0.0, le=1.0)
    valence: float = Field(..., example=0.8, ge=0.0, le=1.0)
    tempo: float = Field(..., example=120.0)
    time_signature: int = Field(..., example=4, ge=3, le=7)
    duration_ms: int = Field(..., example=200000)

@app.post("/predict")
def predict_popularity(track: TrackFeatures):
    try:
        # Создаем DataFrame с фичами в правильном порядке
        features = [
            'danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo', 'time_signature', 'duration_ms'
        ]
        
        input_data = pd.DataFrame([[
            track.danceability, track.energy, track.key, track.loudness,
            track.mode, track.speechiness, track.acousticness,
            track.instrumentalness, track.liveness, track.valence,
            track.tempo, track.time_signature, track.duration_ms
        ]], columns=features)
        
        # Масштабирование фичей
        scaled_data = scaler.transform(input_data)
        
        # Предсказание
        prediction = model.predict(scaled_data)
        
        # Ограничиваем предсказание в диапазоне 0-100
        popularity = np.clip(prediction[0], 0, 100)
        
        return {
            "predicted_popularity": round(float(popularity), 2),
            "feature_importance": dict(zip(
                features,
                model.coef_.tolist()
            ))
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model_info")
def get_model_info():
    """Возвращает информацию о метриках модели"""
    return {
        "model_type": "LinearRegression",
        "features_used": [
            'danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo', 'time_signature', 'duration_ms'
        ],
        "expected_feature_order": model.feature_names_in_.tolist()
    }