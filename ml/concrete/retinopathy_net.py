import torch.nn as nn
import torch

from ml.concrete.residual_block import ResidualBlock


class RetinopathyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)
        self.layer4 = ResidualBlock(128, 256, stride=2)
        self.layer5 = ResidualBlock(256, 256, stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(256, 5)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
