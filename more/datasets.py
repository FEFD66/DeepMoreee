import h5py
import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, transforms
import numpy as np

glabels = ['singlepulse', 'freqdiv', 'LFM', 'QFM', '2FSK', '4FSK', 'BPSK', 'QPSK']


class ModulationDataSets(Dataset):
    __lazy: bool = False
    __PSK: bool = True

    def __init__(self, file, type_select=None):
        self.h5 = h5py.File(file, "r")
        self.types = self.h5['sampleType'][0, :]
        self.type_select = type_select
        self.length = 0
        self.labels = []

        self.toTenser = ToTensor()
        self.img = self.h5['sampleData'][:]
        if type_select is not None:
            mask = self.type_filter(type_select)
            self.img = self.img[:, :, mask]
            self.types = self.types[mask]
            for idx, id in enumerate(type_select):
                self.types[self.types == id] = idx
                self.labels.append(glabels[id-1])
        self.types = torch.tensor(self.types, dtype=torch.long)
        self.length = self.types.shape

    def type_filter(self, types):
        full = (self.types[:] is None)
        for i in types:
            full = full | (self.types[:] == i)
        ids = np.argwhere(full).reshape(-1)
        return ids

    def __getitem__(self, idx):
        return self.toTenser(self.img[:, :, idx]).float(), self.types[idx]

    def __len__(self):
        return self.types.shape[0]

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.h5.close()

    def get_labels_name(self):
        return self.labels

    def get_numclasses(self):
        return 8 if self.type_select is None else len(self.type_select)


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
