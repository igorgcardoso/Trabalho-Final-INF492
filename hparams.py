from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class HParams:
  # Image Params
  size: int = 128

  # Model Params
  input_c: int = 1
  output_c: int = 2
  generator_n_down: int = 8
  num_filters: int = 64
  discriminator_n_down: int = 3

  # Training Params
  batch_size: int = 32
  lr_g: float = 2e-4
  lr_d: float = 2e-4
  betas: Tuple[float] = (0.5, 0.999)
  lambda_L1: float = 100.0
  epochs: int = 50

  # Misc
  device = 'cuda' if torch.cuda.is_available() else 'cpu'