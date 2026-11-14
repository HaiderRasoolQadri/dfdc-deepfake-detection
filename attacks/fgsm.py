import torch
import torch.nn as nn


def fgsm_attack(model, images, labels, epsilon=0.01):
    """
    Untargeted FGSM attack.

    images: tensors in [0, 1]
    labels: ground-truth labels
    epsilon: maximum pixel perturbation
    """
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be between 0 and 1")

    model.eval()

    x = images.detach().clone().requires_grad_(True)

    with torch.enable_grad():
        with torch.autocast(
            device_type="cuda",
            enabled=False
        ):
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, labels)

        gradient = torch.autograd.grad(
            loss, x, only_inputs=True
        )[0]

    adversarial = x + epsilon * gradient.sign()
    adversarial = adversarial.clamp(0.0, 1.0)

    return adversarial.detach()
