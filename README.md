# Spotify Track Popularity Prediction

Проект по развертыванию ML-модели для предсказания популярности треков на Spotify.

## 🎯 Задачи проекта

1. **Экспорт модели через Joblib в .pkl**  
   - Обученная модель линейной регрессии экспортирована в `linear_model.pkl`
   - StandardScaler экспортирован в `scaler.pkl`

2. **Сервис на FastAPI**  
   - REST API для получения предсказаний
   - Входные параметры: 13 аудио-характеристик трека
   - Выход: предсказанная популярность (0-100)

3. **Деплой через Docker**  
   - Контейнеризация приложения
   - Докеризация FastAPI сервиса
   - Оптимизированный Dockerfile

4. **Запуск на Yandex Cloud**  
   - Развертывание на облачной ВМ
   - Использование Yandex Container Registry
   - Настройка сетевых правил для доступа

## 🚀 Быстрый старт

### Запуск через Docker

```bash
docker build -t ml-app .
docker run -p 8000:8000 ml-app
```

API будет доступно на `http://localhost:8000`

## 🌩 Деплой на Yandex Cloud

1. Сборка и пуш образа в Yandex Container Registry
2. Создание ВМ с Docker
3. Запуск контейнера:
```bash
docker run -d -p 80:8000 --name ml-app cr.yandex/<registry-id>/ml-app:latest
```

## 📊 Модель

- Линейная регрессия
- 13 фичей
- R²: 0.65 (на тестовых данных)
- MSE: 112.34
