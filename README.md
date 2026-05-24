# Children Drawings

Классификация детских рисунков по 4 классам и двум дополнительным целям:

- `category`: `house`, `tree`, `man`, `woman`
- `age`: регрессия возраста
- `gender`: `male`, `female`

Проект: **Громаков Илья Алексеевич**

## Постановка задачи

Разработка ML-сервиса для автоматического анализа детского рисунка:

1. определить категорию рисунка (4 класса);
2. оценить возраст автора;
3. оценить пол автора.

Потенциальные применения:

- автоматизация анализа в арт-терапии и детской психологии;
- образовательные инструменты мониторинга развития ребенка;
- построение признаков для исследований связи изобразительных навыков и психологических параметров.

## Формат данных

Вход:

- изображение `jpg/jpeg/png` произвольного размера;
- перед моделью изображение приводится к `300x300x3`.

Выход:

- `class` (строка) + `confidence` (float),
- `age` (int),
- `gender` (строка) + `gender_confidence` (float).

## Метрики

Для основной задачи:

- Accuracy (целевой ориентир `> 0.92`);
- Macro F1-score (целевой ориентир `> 0.92`).

Для дополнительных задач:

- age: MAE;
- gender: Accuracy.

В процессе обучения и валидации логируются `loss`, `accuracy`, `f1`, `mae` по головам модели.

## Датасет

Источник: [ironDong/Children_Drawings](https://huggingface.co/datasets/ironDong/Children_Drawings)

Состав:

- Train: ~40k изображений
- Validation: ~4.5k изображений
- Категории: `집`, `나무`, `남자사람`, `여자사람`

Именование изображений в исходном датасете: `[category]_[age]_[gender]_[id].jpg`.

## Моделирование

Бейзлайн (для сравнения):

- эмбеддинги предобученного ResNet-18 без fine-tune (описан как контрольный вариант).

Основная модель:

- `EfficientNet-B3` (ImageNet pretrained, backbone без classifier),
- дообучение слоев с индексом `> 300`, более ранние слои заморожены,
- три головы: `category`, `age`, `gender`.

Тренировочный стек:

- PyTorch Lightning
- Hydra
- MLflow
- DVC

## Валидация

Текущая реализация использует штатное разделение `train/validation` датасета.
Сид фиксирован для воспроизводимости. K-fold режим остается отдельным расширением.

## Overall (структура проекта)

```text
children_drawings/        # package: data/model/train/evaluate/export/api
conf/                     # Hydra-конфиги (единая точка входа conf/config.yaml)
data/                     # DVC-tracked train/validation/batch
models/                   # Triton model repository + DVC pointer для model.plan
scripts/                  # вспомогательные CLI (download, dvc pull, triton smoke)
docker/                   # Dockerfile'ы
tests/                    # smoke-тесты
compose.yaml              # локальные сервисы: mlflow + triton + web-app
```

## Setup

```bash
uv sync --dev
uv run pre-commit install
uv run dvc pull data/train.dvc data/validation.dvc data/batch.dvc
```

## Train

Базовый запуск:

```bash
uv run children-drawings-train
```

Оценка checkpoint:

```bash
uv run children-drawings-evaluate
```

Пример override:

```bash
uv run children-drawings-train training.epochs=5 data.batch_size=16
```

По умолчанию в основном конфиге установлено `training.epochs=30`.

## Logging (MLflow)

MLflow tracking URI: `http://localhost:8080` (по умолчанию, через конфиг).

Логируется:

- метрики обучения/валидации,
- гиперпараметры запуска,
- `git_commit_id` (версия кода запуска).

Локальный MLflow server:

```bash
docker compose up mlflow
```

## Production Preparation

1. Экспорт checkpoint в ONNX:

```bash
uv run children-drawings-export
```

2. Подтянуть ONNX из DVC (если его нет локально):

```bash
uv run dvc pull artifacts/onnx_models.dvc
```

3. На целевой машине собрать TensorRT engine (`.plan`) из ONNX:

```bash
docker compose run --rm trtexec-build
```

В репозитории хранится только ONNX. Готовый `model.plan` не версионируется и не хранится в DVC.

## DVC: данные и модели

Настроены два remote:

- `r2-storage` — данные (`data/*`)
- `r2-models` — модельные артефакты (`onnx`)

Pull данных:

```bash
uv run dvc pull data/train.dvc data/validation.dvc data/batch.dvc
```

Pull ONNX:

```bash
uv run dvc pull artifacts/onnx_models.dvc
```

Или через вспомогательный CLI:

```bash
uv run python scripts/pull_from_dvc.py onnx
```

## Infer / Serving

Triton + TensorRT:

```bash
docker compose up triton-pipeline
```

Smoke-клиент к Triton:

```bash
uv run python scripts/triton_smoke.py
```

Web-сервис (загрузка файлов + canvas-рисование):

```bash
docker compose up triton-pipeline web-app
```

Endpoints:

- Web UI: `http://localhost:8200`
- Triton HTTP: `http://localhost:8100`
- Triton gRPC: `localhost:8101`
- Triton metrics: `http://localhost:8102/metrics`

## Checks

```bash
uv run pre-commit run -a
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Секреты

Не храните ключи в Git.

Рекомендуемый подход:

1. коммитить только `.dvc/config` и `.dvc/config.local.example`;
2. реальные ключи держать в `.dvc/config.local` (gitignored) или в CI secrets;
3. для локального запуска использовать переменные окружения:
   - `R2_ACCESS_KEY_ID`
   - `R2_SECRET_ACCESS_KEY`.
