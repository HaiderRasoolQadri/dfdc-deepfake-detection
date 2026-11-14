import torch
import torch.nn as nn
import timm


class XceptionDetector(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.backbone = timm.create_model(
            "xception",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )

        for p in self.backbone.parameters():
            p.requires_grad = False

        for name, p in self.backbone.named_parameters():
            if name.startswith((
                "block12", "conv3", "bn3", "conv4", "bn4"
            )):
                p.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 512),
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
