import logging
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from tqdm.contrib.concurrent import process_map

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler('validate_images.log'))


def validate_img(path: Path):
    try:
        img = Image.open(path)
    except UnidentifiedImageError:
        logger.error(f'Could not open {path}')
        path.unlink()
    except OSError:
        logger.error(f'Could not open {path}')
        path.unlink()


imgs = set(Path('data').glob('**/*.jpg'))


process_map(validate_img, imgs, chunksize=128)
