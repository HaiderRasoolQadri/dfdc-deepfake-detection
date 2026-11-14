import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class VGG16Detector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        base = vgg16(
            weights=VGG16_Weights.IMAGENET1K_V1
            if pretrained else None
        )

        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        for p in self.backbone.parameters():
            p.requires_grad = False

        for p in self.backbone[24:].parameters():
            p.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
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
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
