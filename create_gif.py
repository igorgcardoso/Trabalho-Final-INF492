from pathlib import Path

import imageio
import numpy as np

ROOT_DIR = Path(__file__).parent / 'generated'


def get_input(obj_list):
    for idx, obj in enumerate(obj_list):
        print(f"{idx} - {obj.name}")
    return int(input("Select dataset: "))


def get_datasets():
    datasets = list(ROOT_DIR.glob('*'))

    if len(datasets) == 1:
        return datasets[0]

    input_dataset = get_input(datasets)

    return datasets[input_dataset]


def get_runs(dataset: Path):
    runs = list(dataset.glob('*'))

    if len(runs) == 1:
        return runs[0]

    input_run = get_input(runs)

    return runs[input_run]


def get_images(run: Path):
    return list(run.glob('*'))


def main():
    dataset = get_datasets()
    run = get_runs(dataset)
    imgs = get_images(run)

    gif_name = f'{dataset.name}_{run.name}.gif'

    imageio.v3.imwrite(gif_name, np.stack([imageio.v3.imread(img) for img in sorted(imgs)]), axis=0)


if __name__ == '__main__':
    main()
