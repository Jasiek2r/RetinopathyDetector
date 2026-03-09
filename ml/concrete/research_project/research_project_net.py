import torch.nn as nn


class ResearchProjectNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 3x64x64 -> 16x64x64
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x32x32

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 32x32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64x16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x8x8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # 64*8*8 = 4096
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # 5 klas APTOS
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
