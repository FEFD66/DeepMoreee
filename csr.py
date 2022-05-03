import d2l.torch as d2l
from torch.utils.data import Dataset
import torch
import torch.nn as nn

from more import train
from more.datasets import ModulationDataSets, load_data_mnist
from more.network import MoreNet, init_weights
from more.train import train_center
from utils.centerloss import CenterLoss
from utils import get_out_path
import datetime
import time
import os
import platform
from torch.utils.tensorboard import SummaryWriter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

writer = SummaryWriter(log_dir='./log/')
root_path='E:' if platform.system().lower() == 'windows' else '..'
train_path = root_path+r'/train.h5'
test_path = root_path+r'/test.h5'
train_dataset = ModulationDataSets(train_path)
test_dataset = ModulationDataSets(test_path)
label_name = train_dataset.get_labels_name()
num_classes = train_dataset.get_numclasses()
batch_size = 64
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=0, drop_last=False)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=True,
    num_workers=0, drop_last=False)
net = nn.Sequential(
    # 195-1,64-1
    nn.Conv2d(1, 64, kernel_size=2, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
    nn.MaxPool2d((3, 2), (3, 2)),

    nn.Conv2d(64, 128, 3), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(128, 256, 3, stride=2, padding=(0, 1)), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
    nn.AvgPool2d(2, 2),

    nn.Conv2d(256, 1024, 3, stride=2, padding=(0, 1)), nn.BatchNorm2d(1024), nn.ReLU(inplace=True),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(1024, 1024),
    nn.ReLU(inplace=True),
    nn.Linear(1024, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(512, 8),
    nn.Softmax(dim=0)
)
loss = nn.CrossEntropyLoss()
center_loss = CenterLoss(num_classes=num_classes, feat_dim=2, use_gpu=True)
lr = 0.001
lr_center = 0.5
opt_model = torch.optim.SGD(net.parameters(), lr, weight_decay=5e-04, momentum=0.9)
opt_center = torch.optim.SGD(center_loss.parameters(), lr_center)

scheduler = torch.optim.lr_scheduler.StepLR(opt_model, step_size=20, gamma=0.5)

net = net.to("cuda:0")
if os.path.exists('../mod.parm'):
    print('using exits parm')
    net.load_state_dict(torch.load('../mod.parm'))
else:
    print('using random parm')
    init_weights(net)

start_time = time.time()


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)

        writer.add_scalars(main_tag='Metrics', tag_scalar_dict={'train_loss':train_l, 'train_acc':train_acc, 'test_acc': test_acc},global_step=epoch+1)

        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
              f'test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
              f'on {str(device)}')
        if epoch % 10 == 0:
            torch.save(net.state_dict(),'../mod.parm')

train_ch6(net, train_loader, test_loader, 150, 0.01, "cuda:0")
elapsed = round(time.time() - start_time)
elapsed = str(datetime.timedelta(seconds=elapsed))
print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))
for epoch in range(10):
    writer.add_scalar(tag='TrainLoss', scalar_value=epoch / 2, global_step=epoch)
    writer.add_scalars(main_tag='Metrics', tag_scalar_dict={'Loss': 1}, global_step=epoch)

writer.close()
