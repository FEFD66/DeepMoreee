import torch.nn.functional as F
import torch.nn as nn


def old_net():
    return nn.Sequential(
        nn.Conv2d(1, 10, kernel_size=5), nn.MaxPool2d(2), nn.ReLU(),
        nn.Conv2d(10, 20, kernel_size=5), nn.Dropout2d(0.2), nn.MaxPool2d(2), nn.ReLU(),
        nn.Flatten(),
        nn.Linear(11960, 4096), nn.ReLU(),
        nn.Linear(4096, 50), nn.ReLU(),
        nn.Dropout(0),
        nn.Linear(50, 8),
        nn.Softmax()
    )


def new_net():
    return MoreNet()


class MoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(24576, 2))

        self.fc8 = nn.Linear(2, 8)

    def forward(self, x, return_feature=False):
        x = self.conv(x)
        y = self.fc8(x)
        if return_feature:
            return x.detach(), F.softmax(y)
        else:
            return F.softmax(y)
