# Fine-Tuning Lab

Проект решает задачу классификации изображений животных на 3 класса с помощью transfer learning. В репозитории есть подготовка датасета, обучение моделей из `timm`, оценка качества, экспорт лучшей модели в ONNX и локальное Gradio-приложение для CPU-инференса.

## Структура

- `data/raw/` - исходные изображения по классам.
- `data/processed/` - автоматическое разбиение на `train` и `val`.
- `data/README.md` - описание датасета, классов и предобработки.
- `experiments/train.py` - параметризованный скрипт обучения с конфигурацией в дата-классах.
- `experiments/runs/` - результаты отдельных запусков: метрики, графики, матрицы ошибок, ONNX.
- `experiments/models/` - последняя экспортированная модель для приложения.
- `experiments/notebook.ipynb` - ноутбук для сравнения экспериментов и выводов.
- `app/app.py` - Gradio-приложение, использующее `experiments/models/best_model.onnx`.

## Установка

Команды выполняются из корня проекта:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r .\requirements.txt
```

## Подготовка Данных

Изображения должны лежать в подпапках классов:

```text
data/raw/
  class1/
  class2/
  class3/
```

После изменения датасета пересоздайте разбиение:

```powershell
python .\experiments\train.py --force-resplit --epochs 1 --no-pretrained --device cpu
```

Эта команда нужна только для проверки структуры. Для финальных результатов используйте предобученные веса без `--no-pretrained`.

## Обучение Моделей

Первая модель, семейство ResNet:

```powershell
python .\experiments\train.py --run-name resnet18_lr001 --force-resplit --model-name resnet18 --epochs 5 --batch-size 16 --lr 0.001
```

Вторая модель, другое семейство EfficientNet:

```powershell
python .\experiments\train.py --run-name efficientnet_b0_lr001 --model-name efficientnet_b0 --epochs 5 --batch-size 16 --lr 0.001
```

Пример подбора гиперпараметров:

```powershell
python .\experiments\train.py --run-name resnet18_lr0003 --model-name resnet18 --epochs 5 --batch-size 16 --lr 0.0003
python .\experiments\train.py --run-name efficientnet_b0_lr0003 --model-name efficientnet_b0 --epochs 5 --batch-size 16 --lr 0.0003
```

## Результаты

Для каждого запуска создаются файлы:

- `experiments/runs/<run-name>/logs/history.json` - история loss и accuracy по эпохам.
- `experiments/runs/<run-name>/logs/summary.json` - лучшая validation accuracy и параметры запуска.
- `experiments/runs/<run-name>/logs/learning_curves.png` - кривые обучения.
- `experiments/runs/<run-name>/logs/confusion_matrix.png` - матрица ошибок.
- `experiments/runs/<run-name>/models/best_model.onnx` - экспортированная ONNX-модель.
- `experiments/runs/<run-name>/models/classes.json` - порядок классов.

Последний запуск дополнительно копирует модель в `experiments/models/`, откуда ее использует приложение.

## Запуск Приложения

Сначала обучите модель, чтобы появился файл `experiments/models/best_model.onnx`.

```powershell
python .\app\app.py
```

Gradio выведет локальный URL вида `http://127.0.0.1:7860`.

## Воспроизводимость

В `experiments/train.py` фиксируются Python `random`, NumPy и PyTorch. Параметры сгруппированы в дата-классах `DataConfig`, `TrainConfig`, `ArtifactConfig`, а основные значения переопределяются через аргументы командной строки.

Стратегия transfer learning:

- сначала замораживается backbone, обучается только classifier;
- после `--freeze-backbone-epochs` backbone размораживается;
- после разморозки используется уменьшенный learning rate;
- для train используются resize, horizontal flip, rotation и ColorJitter;
- для дисбаланса классов включены class weights в `CrossEntropyLoss`.
