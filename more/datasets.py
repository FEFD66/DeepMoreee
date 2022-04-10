import h5py
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class ModulationDataSets(Dataset):
    __lazy: bool = False
    __PSK: bool = True
    labels = ['singlepulse', 'freqdiv', 'LFM', 'QFM', '2FSK', '4FSK', 'BPSK', 'QPSK']

    def __init__(self, file, lazy=False):
        self.__lazy = lazy
        self.h5 = h5py.File(file, "r")
        self.label_id = torch.tensor(self.h5['sampleType'][:], dtype=torch.long)
        self.length = 0
        _, self.length = self.label_id.shape
        # if self.__PSK:
        #     self.label_id[self.label_id==8]=7

        self.toTenser = ToTensor()
        if not lazy:
            self.img = self.toTenser(self.h5['sampleData'][:]).float().reshape(-1, 1, 196, 64)
        else:
            self.img = self.h5['sampleData']

    def __getitem__(self, idx):
        img = self.img[idx, :, :] if not self.__lazy else self.toTenser(self.img[:, :, idx]).float()
        return img, self.label_id[0, idx] - 1

    def __len__(self):
        return self.length

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.h5.close()

    def get_labels_name(self):
        return self.labels
