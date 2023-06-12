from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm import tqdm

dir = Path(__file__).parent / 'data/downloads'

data = list(dir.glob('**/*.jpg'))

d_train, d_test = train_test_split(data, test_size=0.2)

for file in tqdm(d_train):
  file.rename(file.parent.parent / 'train' / file.name)

for file in tqdm(d_test):
  file.rename(file.parent.parent / 'val' / file.name)