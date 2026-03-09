import torch

from ml.abstractions.ml_engine import MLEngine

import torch.nn as nn
import torch.optim as optim

from ml.concrete.SzUM.szum_project_net import SzumProjectNet
from utility.formatted_date import get_formatted_date


class SzumProjectMLEngine(MLEngine):
    def __init__(self):
        self.__model__ = SzumProjectNet()

    def train(self, x: torch.Tensor, y: torch.Tensor):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.__model__.parameters(), lr=0.001)

        for epoch in range(400):
            optimizer.zero_grad()
            outputs = self.__model__(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, loss = {loss.item():.8f}")
            if(epoch % 100 == 0):
                formatted_date = get_formatted_date()
                print(f"{formatted_date} PLEASE DO NOT TURN OFF THE PROGRAM")

    def test(self, x: torch.Tensor, y: torch.Tensor):
        self.__model__.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            outputs = self.__model__(x)
            _, predicted = torch.max(outputs, 1)

            total = y.size(0)
            correct = (predicted == y).sum().item()

        accuracy = correct / total * 100
        print(f"Test accuracy: {accuracy:.2f}%")
