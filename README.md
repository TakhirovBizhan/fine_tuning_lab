# Fine-Tuning Lab

Проект для дообучения предобученной модели `timm` на локальном наборе изображений, экспорта лучшей модели в ONNX и запуска локального Gradio-приложения для классификации.

## Структура

- `data/raw/` - исходные изображения по классам.
- `data/processed/` - разбиение на `train` и `val`.
- `experiments/train.py` - параметризованный скрипт обучения, оценки и экспорта ONNX.
- `experiments/models/` - чекпойнт, ONNX-модель и список классов после обучения.
- `experiments/logs/` - история обучения, learning curves и confusion matrix.
- `app/app.py` - локальное Gradio-приложение для ONNX-инференса на CPU.

## Установка

Команды ниже выполняются из корня проекта:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r .\requirements.txt
```

## Обучение и экспорт ONNX

Быстрая проверка на 1 эпоху:

```powershell
python .\experiments\train.py --epochs 1
```

Обычный запуск:

```powershell
python .\experiments\train.py --model-name resnet18 --epochs 5 --batch-size 16 --lr 0.001
```

После обучения создаются:

- `experiments/models/best_model.pth`
- `experiments/models/best_model.onnx`
- `experiments/models/classes.json`
- `experiments/logs/history.json`
- `experiments/logs/learning_curves.png`
- `experiments/logs/confusion_matrix.png`

Для второй модели из другого семейства можно запустить, например:

```powershell
python .\experiments\train.py --model-name mobilenetv3_small_100 --epochs 5 --batch-size 16 --lr 0.001
```

## Запуск приложения

Сначала должен быть создан файл `experiments/models/best_model.onnx`.

```powershell
python .\app\app.py
```

Gradio выведет локальный URL вида `http://127.0.0.1:7860`.

## Воспроизводимость

В `experiments/train.py` фиксируются генераторы случайных чисел Python `random`, NumPy и PyTorch. Основные параметры обучения сгруппированы в дата-классах `DataConfig`, `TrainConfig`, `ArtifactConfig` и могут переопределяться через аргументы командной строки.
