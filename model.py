import torch.nn as nn

class SimpleVGG6(nn.Module):
    """Compact 6-layer CNN inspired by VGG."""

    def __init__(self, act_fn):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), act_fn(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), act_fn(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), act_fn(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), act_fn(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
