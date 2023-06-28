from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class HParams:
  # Image Params
  size: int = 128

  # Generator Params
  generator_input_channels: int = 1
  generator_output_channels: int = 2
  generator_num_down: int = 6
  generator_num_filters: int = 64

  # Discriminator Params
  discriminator_input_channels: int = 3
  discriminator_num_layers: int = 3
  discriminator_num_filters_last_layer: int = 32

  # Training Params
  batch_size: int = 16
  lr_generator: float = 2e-4
  lr_discriminator: float = 2e-6
  betas: Tuple[float] = (0.5, 0.999)
  lambda_L1: float = 3.5
  epochs: int = 5

  # Misc
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  visualization_interval: int = 250
