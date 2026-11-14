import json
import random
import time

import numpy as np
import torch
import torch.nn as nn

from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm

from config import MODEL_CONFIGS, CHECKPOINT_DIR, RESULTS_DIR
from dataset import get_dataloaders

from models.xception import XceptionDetector
from models.vgg16 import VGG16Detector
from models.resnet50 import ResNet50Detector
from models.resnet101 import ResNet101Detector


MODEL_CLASSES = {
    "xception": XceptionDetector,
    "vgg16": VGG16Detector,
    "resnet50": ResNet50Detector,
    "resnet101": ResNet101Detector
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_model(name, pretrained=True):
    return MODEL_CLASSES[name](pretrained=pretrained)


def run_epoch(model, loader, criterion, optimizer, scaler, device, training):
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(
        loader,
        desc="Training" if training else "Validation"
    )

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with autocast(
                device_type="cuda",
                enabled=(device.type == "cuda")
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = labels.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(1) == labels).sum().item()

        progress.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            accuracy=f"{total_correct / total_samples:.4f}"
        )

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples
    }


def train_model(name, epochs=1, resume=False):
    set_seed()

    config = MODEL_CONFIGS[name]
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    loaders = get_dataloaders(config["batch_size"])
    model = create_model(name, pretrained=not resume).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    scaler = GradScaler(
        "cuda",
        enabled=(device.type == "cuda")
    )

    scheduler = None

    if name == "xception":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.5
        )

    checkpoint_path = CHECKPOINT_DIR / f"{name}_latest.pth"
    best_path = CHECKPOINT_DIR / f"{name}_best.pth"
    history_path = RESULTS_DIR / f"{name}_history.json"

    start_epoch = 1
    best_loss = float("inf")
    patience_counter = 0
    history = []

    if resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_val_loss"]
        patience_counter = checkpoint.get("patience_counter", 0)
        history = checkpoint.get("history", [])

    for epoch in range(start_epoch, epochs + 1):
        started = time.time()

        print(f"\n{name.upper()} | Epoch {epoch}/{epochs}")

        train_metrics = run_epoch(
            model, loaders["train"], criterion,
            optimizer, scaler, device, True
        )

        valid_metrics = run_epoch(
            model, loaders["valid"], criterion,
            None, scaler, device, False
        )

        if scheduler:
            scheduler.step()

        record = {
            "model": name,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": valid_metrics["loss"],
            "val_accuracy": valid_metrics["accuracy"],
            "minutes": (time.time() - started) / 60
        }

        history.append(record)

        improved = valid_metrics["loss"] < best_loss

        if improved:
            best_loss = valid_metrics["loss"]
            patience_counter = 0
        else:
            patience_counter += 1

        state = {
            "model_name": name,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler else None
            ),
            "best_val_loss": best_loss,
            "patience_counter": patience_counter,
            "history": history
        }

        torch.save(state, checkpoint_path)

        if improved:
            torch.save(state, best_path)

        history_path.write_text(
            json.dumps(history, indent=2)
        )

        print(
            f"Train: loss={train_metrics['loss']:.4f}, "
            f"acc={train_metrics['accuracy']:.4%}"
        )
        print(
            f"Valid: loss={valid_metrics['loss']:.4f}, "
            f"acc={valid_metrics['accuracy']:.4%}"
        )
        print(f"Time: {record['minutes']:.2f} min")

        if patience_counter >= config["patience"]:
            print("Early stopping triggered")
            break

    del model, optimizer, scaler

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return history
