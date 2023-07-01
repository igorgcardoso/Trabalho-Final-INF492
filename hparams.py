from dataclasses import dataclass
from typing import Literal, Tuple

import torch


@dataclass
class HParams:
  # Image Params
  size: int = 128

  # Generator Params
  generator_input_channels: int = 1
  generator_output_channels: int = 2
  generator_num_down: int = 7
  generator_num_filters: int = 96

  # Discriminator Params
  discriminator_input_channels: int = 3
  discriminator_num_layers: int = 3
  discriminator_num_filters_last_layer: int = 32

  # Training Params
  batch_size: int = 8
  lr_generator: float = 2e-5
  lr_discriminator: float = 2e-5
  betas: Tuple[float] = (0.5, 0.999)
  lambda_L1: float = 2.0
  epochs: int = 25

  # Misc
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  visualization_interval: int = 100
  dataset: Literal['colorful_colorization', 'unplash', 'imagenette', 'imagewoof', 'fruits'] = 'fruits'
