import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from torch.amp import autocast

from config import CHECKPOINT_DIR
from trainer import create_model


def load_trained_model(name, device):
    model = create_model(name, pretrained=False)

    checkpoint = torch.load(
        CHECKPOINT_DIR / f"{name}_best.pth",
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


@torch.inference_mode()
def predict_probabilities(model, loader, device):
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)

        with autocast(
            device_type="cuda",
            enabled=(device.type == "cuda")
        ):
            logits = model(images)

        probabilities = torch.softmax(
            logits.float(), dim=1
        )[:, 1]

        all_probs.extend(probabilities.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_probs)


def calculate_metrics(labels, probabilities):
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(
            labels, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            labels, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            labels, predictions, zero_division=0
        )),
        "roc_auc": float(roc_auc_score(
            labels, probabilities
        )),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist()
    }
