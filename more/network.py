import torch.nn.functional as F
import torch.nn as nn


def old_conv(feat_dim):
    return nn.Sequential(
        nn.Conv2d(1, 10, kernel_size=5), nn.MaxPool2d(2), nn.ReLU(),
        nn.Conv2d(10, 20, kernel_size=5), nn.Dropout2d(0.2), nn.MaxPool2d(2), nn.ReLU(),
        nn.Flatten(),
        nn.Linear(11700, 4096), nn.ReLU(),
        nn.Linear(4096, feat_dim)
    )


def new_conv():
    return nn.Sequential(
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
        nn.Linear(24576, 4096), nn.ReLU(),
        nn.Linear(4096, 2), nn.ReLU())


def new_net():
    return MoreNet()


class MoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = old_conv(2)
        # self.fc_feat = nn.Linear(128, 2)
        self.fc8 = nn.Linear(2, 8)

    def forward(self, x, return_feature=False):
        x = self.conv(x)
        # x = self.fc_feat(x)
        y = self.fc8(x)
        if return_feature:
            # return x, F.softmax(y, dim=0)
            return x,y
        else:
            return F.softmax(y, dim=0)


class ARPLNet(nn.Module):
    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.conv = old_conv(feat_dim)
        self.fc = nn.Linear(feat_dim, num_classes, bias=False)

    def forward(self, x, return_feature=False):
        x = self.conv(x)
        y = self.fc(F.relu(x))
        if return_feature:
            return x, y
        else:
            return y


def init_weights(net):
    def init(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init)
