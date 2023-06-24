import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import functional as TF


def salt_and_pepper(img: torch.Tensor | Image.Image, prob: float = 0.05):
    """
    Apply salt and pepper noise to the input image.

    Args:
        img (PIL Image or tensor): Input image.
        prob (float): Probability of each pixel being affected by the noise.
    Returns:
        PIL Image or tensor: Noisy image.
    """
    img = TF.to_tensor(img) if isinstance(img, Image.Image) else img
    img = img.clone()

    if np.random.random() < prob:
        width, height = img.shape[-1], img.shape[-2]

        num_pixels = width * height

        num_noise_pixels = int(num_pixels * prob)

        for _ in range(num_noise_pixels):
            x = np.random.randint(0, width - 1)
            y = np.random.randint(0, height - 1)
            value = np.random.choice([0, 1])
            img[..., y, x] = value
    return img


class GaussianNoise(nn.Module):
    def __init__(self, mean=0., std=.4):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, x):
        return x + torch.randn(x.size()) * self.std + self.mean
