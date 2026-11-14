import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights


class ResNet101Detector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.backbone = resnet101(
            weights=ResNet101_Weights.IMAGENET1K_V2
            if pretrained else None
        )

        features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        for p in self.backbone.parameters():
            p.requires_grad = False

        for p in self.backbone.layer4.parameters():
            p.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2)
        )

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.classifier(self.backbone(x))
