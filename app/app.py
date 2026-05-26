import json
from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Скрипт обучения копирует сюда последние/лучшие экспортированные артефакты.
# app.py не использует PyTorch: ONNX-модель загружается через onnxruntime.
MODEL_PATH = PROJECT_ROOT / "experiments" / "models" / "best_model.onnx"
CLASSES_PATH = PROJECT_ROOT / "experiments" / "models" / "classes.json"

# Эти значения должны совпадать с validation preprocessing в experiments/train.py.
IMAGE_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# Если обучение/экспорт еще не запускались, сразу показываем понятную ошибку.
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"ONNX model not found: {MODEL_PATH}. Run `python experiments/train.py` first."
    )
if not CLASSES_PATH.exists():
    raise FileNotFoundError(
        f"Class mapping not found: {CLASSES_PATH}. Run `python experiments/train.py` first."
    )


with CLASSES_PATH.open("r", encoding="utf-8") as file:
    CLASSES = json.load(file)

# CPUExecutionProvider гарантирует локальный запуск демо без видеокарты.
SESSION = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
INPUT_NAME = SESSION.get_inputs()[0].name


def preprocess(image: Image.Image) -> np.ndarray:
    """Применяет такую же детерминированную обработку, как на validation.

    Вход ONNX-модели имеет форму [batch, channels, height, width], поэтому
    изображение переводится из HWC (Pillow/NumPy) в CHW, а затем добавляется
    batch-измерение.
    """

    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    return np.transpose(array, (2, 0, 1))[None, ...].astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Преобразует сырые logits модели в вероятности классов."""

    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def predict(image: Image.Image) -> dict[str, float]:
    """Запускает ONNX-инференс и возвращает вероятности для Gradio Label."""

    inputs = preprocess(image)
    logits = SESSION.run(None, {INPUT_NAME: inputs})[0]
    probabilities = softmax(logits)[0]
    return {
        class_name: float(probability)
        for class_name, probability in zip(CLASSES, probabilities)
    }


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Image"),
    outputs=gr.Label(num_top_classes=len(CLASSES), label="Prediction"),
    title="Animal Classifier",
)


if __name__ == "__main__":
    demo.launch()
