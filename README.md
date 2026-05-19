# Children Drawings

Классификация детских рисунков с несколькими головами модели:

- `category`: дом, дерево, мужчина, женщина;
- `age`: регрессия возраста;
- `gender`: классификация пола.

Проект использует PyTorch Lightning, Hydra, DVC, MLflow и ONNXRuntime.

## Быстрый старт

```bash
uv sync --dev
```

Данные управляются через DVC. Публичная часть remote лежит в `.dvc/config`, а
ключи должны быть только в `.dvc/config.local`:

```bash
uv run dvc pull
```

## Команды

Обучение:

```bash
uv run children-drawings-train
```

Оценка checkpoint:

```bash
uv run children-drawings-evaluate
```

Экспорт checkpoint в ONNX и проверка parity:

```bash
uv run children-drawings-export
```

PyTorch inference по изображениям из `conf/inference/default.yaml`:

```bash
uv run children-drawings-infer
```

FastAPI inference поверх ONNXRuntime:

```bash
uv run children-drawings-api
```

После запуска API:

```bash
curl http://localhost:8000/health
curl -F "file=@data/batch/example.png" http://localhost:8000/predict
```

## Конфигурация

Основной конфиг: `conf/config.yaml`.

Полезные override-примеры:

```bash
uv run children-drawings-train training.epochs=5 data.batch_size=16
uv run children-drawings-export inference.best=best.ckpt
uv run children-drawings-infer inference.images=data/batch
```

MLflow tracking URI берется из `conf/logger/default.yaml` или из переменной:

```bash
export MLFLOW_TRACKING_URI=http://localhost:8080
```

## Docker и MLflow

MLflow Tracking Server:

```bash
docker compose up mlflow
```

Подтянуть DVC-данные в локальные volume/папки:

```bash
docker compose --profile data run --rm dvc-pull
```

Обучение в Docker на локальной видеокарте:

```bash
docker compose --profile train run --rm --gpus all trainer
```

Для GPU внутри контейнера нужна локальная NVIDIA-видеокарта, драйвер на хосте
и NVIDIA Container Toolkit. Видеокарта не арендуется: контейнер получает доступ
к устройству хоста через `--gpus all`.

ONNX API:

```bash
docker compose up api
```

## Проверки

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python -m compileall children_drawings scripts
```

Smoke-тесты проверяют загрузку Hydra-конфига, dataloader на временном датасете,
forward pass модели без pretrained-весов, ONNX export/parity и регистрацию API.

## Артефакты

- Checkpoint: `artifacts/checkpoints/best.ckpt`
- ONNX: `artifacts/onnx_models/children_drawings.onnx`
- External ONNX weights: `artifacts/onnx_models/children_drawings.onnx.data`
- MLflow: Docker volume `mlflow-data` или локальная папка при локальном запуске

Важно: для деплоя ONNX нужны оба файла: `.onnx` и `.onnx.data`.
