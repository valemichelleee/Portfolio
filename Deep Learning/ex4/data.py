from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode

        self._transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_path = self.data.iloc[index, 0]
        label = self.data.iloc[index, 1:]

        image = imread(img_path)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        label = np.array([label], dtype=float).reshape(-1, 2)

        if self._transform:
            image = self._transform(image)

        return image, label

