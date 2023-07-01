from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageFile
from skimage import color
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from hparams import HParams

data_dir = Path(__file__).parent / 'data'

DATASETS = [dataset.name for dataset in data_dir.glob('*')]

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageDataset(Dataset):
    def __init__(self, split: Literal['train', 'val'] = 'train'):

        self.split = split
        self.size = HParams.size

        dataset = HParams.dataset

        if dataset not in DATASETS:
            raise ValueError(f'Invalid dataset: {dataset}')

        if split == 'train':
            self.data_dir = data_dir / f'{dataset}/train'
        elif split == 'val':
            self.data_dir = data_dir / f'{dataset}/val'
        else:
            raise ValueError(f'Invalid split: {split}')

        self.transforms = transforms.Compose([
            transforms.Resize((HParams.size, HParams.size), transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
        ])

        self.data = list(self.data_dir.glob('*'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        img = self.transforms(img)
        img = np.array(img)
        img_lab = color.rgb2lab(img).astype(np.float32)
        lab_tensor = transforms.ToTensor()(img_lab)

        L = lab_tensor[[0], ...] / 50.0 - 1.0
        AB = lab_tensor[[1, 2], ...] / 110.0

        return {'A': L, 'B': AB}


def make_dataloaders():
    train_dataset = ImageDataset(split='train')
    val_dataset = ImageDataset(split='val')

    train_loader = DataLoader(train_dataset, batch_size=HParams.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=HParams.batch_size, shuffle=False)

    return train_loader, val_loader
