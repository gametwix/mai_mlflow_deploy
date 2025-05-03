
# MLOPS Project: Развертывание ML модели через MLFlow

Проект демонстрирует процесс упаковки модели YOLO в MLFlow.

## Структура проекта
```
.
├── .gitignore
├── yolo_mlflow.ipynb            # Основной ноутбук с реализацией
├── yolo_model_mlflow_wrapper.py # Кастомная обертка для MLFlow
├── pyproject.toml               # Конфигурация зависимостей Poetry
├── poetry.lock                  # Lock-файл зависимостей
└── README.md                    # Этот файл
```

## 📋 Требования
- Python 3.10+
- Poetry (для управления зависимостями)
- MLFlow Server
- Docker (для контейнеризации)

## ⚙️ Установка
1. Клонировать репозиторий
2. Установить зависимости:
```bash
poetry install
```

## 🚀 Основные шаги

### 1. Обучение и логирование модели
Запустите ноутбук `yolo_mlflow.ipynb` для:
- Загрузки тестового изображения
- Тестирования YOLO локально
- Создания MLFlow-обертки
- Логирования модели в MLFlow Registry

### 2. Запуск MLFlow Server
```bash
mlflow server --host 0.0.0.0 --port 8098
```

### 3. Тестирование модели локально
```python
mlflow.models.predict(
    model_uri="runs:/<RUN_ID>/model",
    input_data=np.array(img),
    env_manager="conda"
)
```

### 4. Сборка Docker-образа
```bash
mlflow models build-docker -m "runs:/<RUN_ID>/model" -n "mlflow_yolo"
```

### 5. Запуск контейнера
```bash
docker run -p 5000:8080 -it --rm mlflow_yolo
```

## 🌐 Пример запроса к API
```python
import requests
import base64

headers = {"Content-Type": "application/json"}
img_base64 = base64.b64encode(img_bytes).decode("utf-8")

response = requests.post(
    "http://localhost:5000/invocations",
    headers=headers,
    json={"inputs": img_base64}
)
```