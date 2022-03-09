import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

name = ['singlepulse', 'freqdiv', 'LFM', 'QFM', '2FSK', '4FSK', 'BPSK', 'QPSK']


class ModulationDataSets(Dataset):
    def __init__(self, file):
        self.h5 = h5py.File(file, "r")
        self.label_id = self.h5['sampleType']
        _, self.length = self.label_id.shape
        self.img = self.h5['sampleData']
        self.toTensor = ToTensor()

    def __getitem__(self, idx):
        return self.toTensor(self.img[:, :, idx]).float(), torch.tensor(self.label_id[0, idx] - 1, dtype=torch.long)

    def __len__(self):
        return self.length

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.h5.close()


X = torch.randn(size=(1, 1, 196, 64))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__, 'output shape:\t', X.shape)

net.load_state_dict(torch.load('vgg_11.parm'))

test_path = r'E:\test.h5'

test_dataset = ModulationDataSets(test_path)
batch_size = 100
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size,
    shuffle=False, num_workers=0, drop_last=False)
## 提取单独类型

net.eval()
for img, labels in test_loader:
    output = net(img)
    pred = output.data.max(1, keepdim=True)[1].reshape(100)
    pred = pred.numpy()
    labels = labels.reshape(100).numpy()
    err_idx = np.argwhere(pred != labels)
    print("{} accuracy:{}".format(name[labels[0]], 100 - 100 * len(err_idx) / 100))
    for i in (err_idx):
        i = i[0]
        plt.imshow(img[i, 0, :, :].t(), cmap="gray")
        plt.title('predict:{} real:{}'.format(name[pred[i]], name[labels[i]]))
        plt.show()
    pass
