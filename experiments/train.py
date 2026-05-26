import argparse
import json
import random
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import timm
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class DataConfig:
    """Настройки, связанные с датасетом.

    Хранение параметров в дата-классе упрощает повторение эксперимента:
    одни и те же пути, пропорция train/val и размер изображения используются
    повторно, а через CLI меняются только нужные значения.
    """

    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    train_ratio: float = 0.8
    image_size: int = 224
    force_resplit: bool = False
    valid_exts: tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    )


@dataclass
class TrainConfig:
    """Гиперпараметры обучения и настройки воспроизводимости."""

    model_name: str = "resnet18"
    pretrained: bool = True
    batch_size: int = 16
    lr: float = 1e-3
    epochs: int = 5
    seed: int = 42
    num_workers: int = 0
    # Сначала обучается только новая классификационная голова, затем вся модель.
    freeze_backbone_epochs: int = 2
    # После разморозки LR уменьшается, чтобы не разрушить предобученные признаки.
    unfreeze_lr_factor: float = 0.1
    # Помогает при дисбалансе классов, когда изображений в классах разное число.
    use_class_weights: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ArtifactConfig:
    """Пути и имена файлов для сохранения результатов эксперимента."""

    output_dir: Path = PROJECT_ROOT / "experiments" / "models"
    log_dir: Path = PROJECT_ROOT / "experiments" / "logs"
    deploy_dir: Path = PROJECT_ROOT / "experiments" / "models"
    checkpoint_name: str = "best_model.pth"
    onnx_name: str = "best_model.onnx"
    classes_name: str = "classes.json"


@dataclass
class ExperimentConfig:
    """Общая конфигурация: данные, обучение и выходные артефакты."""

    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)


def set_seed(seed: int) -> None:
    """Фиксирует генераторы случайных чисел для повторяемости запусков."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def config_to_jsonable(config: ExperimentConfig) -> dict:
    """Преобразует конфигурацию в JSON-совместимый вид для summary/checkpoint."""

    payload = {
        "data": asdict(config.data),
        "train": asdict(config.train),
        "artifacts": asdict(config.artifacts),
    }
    for section in payload.values():
        for key, value in list(section.items()):
            if isinstance(value, Path):
                section[key] = str(value)
    return payload


def prepare_data(config: DataConfig, seed: int) -> None:
    """Создает train/val из data/raw, если data/processed еще не подготовлен.

    data/raw должен соответствовать формату ImageFolder:
        data/raw/class1/*.jpg
        data/raw/class2/*.jpg
        data/raw/class3/*.jpg

    Если передан --force-resplit, старое разбиение удаляется и создается заново.
    Это важно после изменения датасета, иначе обучение может взять старые
    изображения из data/processed.
    """

    if config.force_resplit and config.processed_data_dir.exists():
        shutil.rmtree(config.processed_data_dir)

    train_dir = config.processed_data_dir / "train"
    val_dir = config.processed_data_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    has_processed_images = any(
        path.is_file() and path.suffix.lower() in config.valid_exts
        for path in config.processed_data_dir.rglob("*")
    )
    if has_processed_images:
        print("[OK] data/processed already contains images; split is reused.")
        return

    # Локальный RNG с фиксированным seed делает разбиение train/val повторяемым.
    rng = random.Random(seed)
    for class_dir in sorted(path for path in config.raw_data_dir.iterdir() if path.is_dir()):
        images = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in config.valid_exts
        )
        if not images:
            continue

        rng.shuffle(images)
        split_index = int(config.train_ratio * len(images))
        for split_name, split_images in (
            ("train", images[:split_index]),
            ("val", images[split_index:]),
        ):
            target_dir = config.processed_data_dir / split_name / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            for image_path in split_images:
                shutil.copy2(image_path, target_dir / image_path.name)

    print("[OK] data/raw was split into data/processed/train and data/processed/val.")


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Создает preprocessing для обучения и валидации.

    Train использует аугментации, чтобы модель была устойчивее к изменениям
    изображений. Validation использует только детерминированную обработку,
    чтобы метрики считались на стабильных, неизмененных данных.
    """

    # Нормализация ImageNet нужна, потому что предобученные timm-модели
    # обучались именно на таком распределении входных изображений.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tfms, val_tfms


def make_loaders(config: ExperimentConfig) -> tuple[DataLoader, DataLoader, list[str]]:
    """Создает ImageFolder-датасеты и DataLoader-ы."""

    train_tfms, val_tfms = build_transforms(config.data.image_size)
    train_dataset = datasets.ImageFolder(
        config.data.processed_data_dir / "train", transform=train_tfms
    )
    val_dataset = datasets.ImageFolder(
        config.data.processed_data_dir / "val", transform=val_tfms
    )

    # Генератор с seed делает перемешивание DataLoader повторяемым.
    generator = torch.Generator().manual_seed(config.train.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )
    return train_loader, val_loader, train_dataset.classes


def dataset_class_counts(dataset: datasets.ImageFolder, num_classes: int) -> list[int]:
    """Считает количество изображений в каждом классе train-датасета."""

    counts = [0] * num_classes
    for label in dataset.targets:
        counts[label] += 1
    return counts


def make_class_weights(counts: list[int], device: str) -> torch.Tensor:
    """Считает веса классов для CrossEntropyLoss.

    Классы с меньшим числом изображений получают больший вес, поэтому ошибки
    на них сильнее штрафуются во время обучения.
    """

    total = sum(counts)
    num_classes = len(counts)
    weights = [total / (num_classes * count) if count else 0.0 for count in counts]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    """Создает timm-модель и заменяет classifier под наши классы.

    Этап freeze:
    - сначала замораживаются все параметры модели;
    - обучаемой остается только новая классификационная голова.

    Это transfer learning: backbone сохраняет полезные признаки ImageNet,
    а classifier учится сопоставлять эти признаки с классами лабораторной.
    """

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.get_classifier().parameters():
        parameter.requires_grad = True
    return model


def unfreeze_model(model: nn.Module) -> None:
    """Разрешает обновлять все веса модели на этапе fine-tuning."""

    for parameter in model.parameters():
        parameter.requires_grad = True


def make_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    """Оптимизирует только параметры, у которых requires_grad=True."""

    return torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=lr,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    """Выполняет одну эпоху обучения и возвращает loss и accuracy."""

    model.train()
    losses: list[float] = []
    targets: list[int] = []
    predictions: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(outputs.argmax(dim=1).detach().cpu().tolist())

    return float(np.mean(losses)), accuracy_score(targets, predictions)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str
) -> tuple[float, float, list[int], list[int]]:
    """Оценивает модель без градиентов и возвращает метрики/предсказания."""

    model.eval()
    losses: list[float] = []
    targets: list[int] = []
    predictions: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        losses.append(loss.item())
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(outputs.argmax(dim=1).detach().cpu().tolist())

    return float(np.mean(losses)), accuracy_score(targets, predictions), targets, predictions


def save_plots(
    history: list[dict[str, float]],
    targets: list[int],
    predictions: list[int],
    classes: list[str],
    artifact_config: ArtifactConfig,
) -> None:
    """Сохраняет learning curves и confusion matrix для анализа в отчете."""

    artifact_config.log_dir.mkdir(parents=True, exist_ok=True)

    epochs = [item["epoch"] for item in history]
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, [item["train_loss"] for item in history], label="train loss")
    plt.plot(epochs, [item["val_loss"] for item in history], label="val loss")
    plt.plot(epochs, [item["train_acc"] for item in history], label="train acc")
    plt.plot(epochs, [item["val_acc"] for item in history], label="val acc")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifact_config.log_dir / "learning_curves.png")
    plt.close()

    matrix = confusion_matrix(targets, predictions, labels=list(range(len(classes))))
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        xticklabels=classes,
        yticklabels=classes,
        cmap="Blues",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(artifact_config.log_dir / "confusion_matrix.png")
    plt.close()


def export_onnx(
    model: nn.Module,
    config: ExperimentConfig,
) -> None:
    """Экспортирует лучшую PyTorch-модель в ONNX для CPU-инференса в app."""

    model.eval().cpu()
    # Dummy input задает форму тензора, которую ожидает экспортируемая модель.
    dummy_input = torch.randn(1, 3, config.data.image_size, config.data.image_size)
    output_path = config.artifacts.output_dir / config.artifacts.onnx_name
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    print(f"[OK] ONNX exported: {output_path}")


def sync_app_artifacts(config: ExperimentConfig) -> None:
    """Копирует последние артефакты в experiments/models для app.py."""

    config.artifacts.deploy_dir.mkdir(parents=True, exist_ok=True)
    for file_name in (
        config.artifacts.onnx_name,
        config.artifacts.classes_name,
        config.artifacts.checkpoint_name,
    ):
        source = config.artifacts.output_dir / file_name
        if source.exists():
            shutil.copy2(source, config.artifacts.deploy_dir / file_name)


def run_training(config: ExperimentConfig) -> None:
    """Основной pipeline: split данных, обучение, оценка, сохранение и экспорт."""

    set_seed(config.train.seed)
    config.artifacts.output_dir.mkdir(parents=True, exist_ok=True)
    config.artifacts.log_dir.mkdir(parents=True, exist_ok=True)

    prepare_data(config.data, config.train.seed)
    train_loader, val_loader, classes = make_loaders(config)
    num_classes = len(classes)
    class_counts = dataset_class_counts(train_loader.dataset, num_classes)

    model = build_model(config.train.model_name, num_classes, config.train.pretrained).to(
        config.train.device
    )
    class_weights = (
        make_class_weights(class_counts, config.train.device)
        if config.train.use_class_weights
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = make_optimizer(model, config.train.lr)

    best_val_acc = -1.0
    best_targets: list[int] = []
    best_predictions: list[int] = []
    history: list[dict[str, float]] = []
    checkpoint_path = config.artifacts.output_dir / config.artifacts.checkpoint_name

    for epoch in range(1, config.train.epochs + 1):
        if epoch == config.train.freeze_backbone_epochs + 1:
            # Этап unfreeze: теперь backbone тоже обучается, но с меньшим LR.
            unfreeze_model(model)
            optimizer = make_optimizer(model, config.train.lr * config.train.unfreeze_lr_factor)
            print("[OK] Backbone unfrozen for fine-tuning.")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.train.device
        )
        val_loss, val_acc, targets, predictions = evaluate(
            model, val_loader, criterion, config.train.device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        print(
            f"Epoch {epoch:02d}/{config.train.epochs}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            # Сохраняем только checkpoint с лучшей validation accuracy.
            best_val_acc = val_acc
            best_targets = targets
            best_predictions = predictions
            torch.save(
                {
                    "model_name": config.train.model_name,
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "config": config_to_jsonable(config),
                    "best_val_acc": best_val_acc,
                },
                checkpoint_path,
            )

    with (config.artifacts.log_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)
    with (config.artifacts.output_dir / config.artifacts.classes_name).open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(classes, file, ensure_ascii=False, indent=2)

    save_plots(history, best_targets, best_predictions, classes, config.artifacts)
    summary = {
        "model_name": config.train.model_name,
        "pretrained": config.train.pretrained,
        "epochs": config.train.epochs,
        "batch_size": config.train.batch_size,
        "lr": config.train.lr,
        "freeze_backbone_epochs": config.train.freeze_backbone_epochs,
        "use_class_weights": config.train.use_class_weights,
        "classes": classes,
        "train_class_counts": dict(zip(classes, class_counts)),
        "best_val_acc": best_val_acc,
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "onnx": str(config.artifacts.output_dir / config.artifacts.onnx_name),
            "learning_curves": str(config.artifacts.log_dir / "learning_curves.png"),
            "confusion_matrix": str(config.artifacts.log_dir / "confusion_matrix.png"),
        },
    }
    with (config.artifacts.log_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    export_onnx(model, config)
    sync_app_artifacts(config)
    print(f"[OK] Best validation accuracy: {best_val_acc:.4f}")
    print(f"[OK] App artifacts updated in: {config.artifacts.deploy_dir}")


def parse_args() -> argparse.Namespace:
    """CLI-аргументы для повторения и изменения экспериментов из терминала."""

    parser = argparse.ArgumentParser(description="Train timm classifier and export ONNX.")
    parser.add_argument("--model-name", default=TrainConfig.model_name)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", default=TrainConfig.device)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--force-resplit", action="store_true")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=TrainConfig.freeze_backbone_epochs)
    return parser.parse_args()


def main() -> None:
    """Собирает config из значений по умолчанию и CLI, затем запускает training."""

    args = parse_args()
    config = ExperimentConfig()
    config.train.model_name = args.model_name
    config.train.epochs = args.epochs
    config.train.batch_size = args.batch_size
    config.train.lr = args.lr
    config.train.seed = args.seed
    config.train.device = args.device
    config.train.pretrained = not args.no_pretrained
    config.train.use_class_weights = not args.no_class_weights
    config.train.freeze_backbone_epochs = args.freeze_backbone_epochs
    config.data.force_resplit = args.force_resplit
    if args.run_name:
        # run-name не дает разным экспериментам перезаписывать друг друга.
        run_dir = PROJECT_ROOT / "experiments" / "runs" / args.run_name
        config.artifacts.output_dir = run_dir / "models"
        config.artifacts.log_dir = run_dir / "logs"
    run_training(config)


if __name__ == "__main__":
    main()
