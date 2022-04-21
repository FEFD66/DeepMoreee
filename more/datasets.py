import h5py
import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, transforms
import numpy as np


class ModulationDataSets(Dataset):
    __lazy: bool = False
    __PSK: bool = True
    labels = ['singlepulse', 'freqdiv', 'LFM', 'QFM', '2FSK', '4FSK', 'BPSK', 'QPSK']

    def __init__(self, file, type_select=None):
        self.h5 = h5py.File(file, "r")
        self.label_id = self.h5['sampleType'][:]
        self.type_select = type_select
        self.length = 0
        _, self.length = self.label_id.shape

        self.toTenser = ToTensor()
        self.img = self.h5['sampleData'][:]
        mask = self.type_filter(type_select)
        if type_select is not None:
            self.img = self.img[:, :, mask]
        self.img = self.toTenser(self.img)
        self.label_id = torch.tensor(self.label_id[:, mask], dtype=torch.long)

    def type_filter(self, types):
        full = (self.label_id[0, :] is None)
        for i in types:
            full = full | (self.label_id[0, :] == i)
        ids = np.argwhere(full).reshape(-1)
        return ids

    def __getitem__(self, idx):
        img = self.img[idx, :, :]  # if not self.__lazy else self.toTenser(self.img[:, :, idx]).float()
        return img, self.label_id[0, idx] - 1

    def __len__(self):
        return self.length

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.h5.close()

    def get_labels_name(self):
        return self.labels

    def get_numclasses(self):
        return len(self.labels[self.type_select])


def load_more_data(path: str, batch_size: int = 64) -> (DataLoader, DataLoader):
    train_path = path + '/train.h5'
    test_path = path + '/test.h5'
    train_dataset = ModulationDataSets(train_path)
    test_dataset = ModulationDataSets(test_path)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=False)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=False)
    return train_loader, test_loader


def load_osr_data(path: str, known, unknown, batch_size: int = 64) -> (DataLoader, DataLoader, DataLoader):
    train_path = path + '/train.h5'
    test_path = path + '/test.h5'
    train_dataset = ModulationDataSets(train_path, type_select=known)
    test_dataset = ModulationDataSets(test_path, type_select=known)
    out_dataset = ModulationDataSets(test_path, type_select=unknown)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=False)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=False)
    out_loader = DataLoader(
        dataset=out_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, drop_last=False)
    return train_loader, test_loader, out_loader
    pass


def load_data_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.MNIST(
        root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.MNIST(
        root="./data", train=False, transform=trans, download=True)
    return (torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                        num_workers=0),
            torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                        num_workers=0))
