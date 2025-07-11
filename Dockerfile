FROM python:3.9-slim

# Устанавливаем зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi uvicorn joblib numpy scikit-learn python-multipart

# Копируем файлы
COPY linear_model.pkl scaler.pkl app.py ./

# Запускаем сервер
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]