import numpy as np


def average_ensemble(resnet50_probs, resnet101_probs):
    if len(resnet50_probs) != len(resnet101_probs):
        raise ValueError("Prediction lengths do not match")

    return (
        np.asarray(resnet50_probs) +
        np.asarray(resnet101_probs)
    ) / 2.0
