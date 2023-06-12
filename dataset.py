from pathlib import Path
from typing import Literal

import numpy as np
from hparams import HParams
from PIL import Image
from skimage.color import rgb2lab
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

data_dir = Path(__file__).parent / 'data'

class ImageDataset(Dataset):
  def __init__(self, split: Literal['train', 'val'] = 'train'):

    self.split = split
    self.size = HParams.size

    if split == 'train':
      self.data_dir = Path(__file__).parent / 'data/train'
      self.transforms = transforms.Compose([
        transforms.Resize((HParams.size, HParams.size), Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
      ])
    elif split == 'val':
      self.data_dir = Path(__file__).parent / 'data/val'
      self.transforms = transforms.Resize((HParams.size, HParams.size), Image.BICUBIC)
    else:
      raise ValueError('mode must be either "train" or "val"')

    self.data = list(self.data_dir.glob('*.jpg'))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    img = Image.open(self.data[idx]).convert('RGB')
    img = self.transforms(img)
    img = np.array(img)
    img_lab = rgb2lab(img).astype("float32")
    img_lab = transforms.ToTensor()(img_lab)

    L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
    ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1

    return {'L': L, 'ab': ab}


def make_dataloaders():
  train_dataset = ImageDataset(split='train')
  val_dataset = ImageDataset(split='val')

  train_loader = DataLoader(train_dataset, batch_size=HParams.batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=HParams.batch_size, shuffle=False)

  return train_loader, val_loader
