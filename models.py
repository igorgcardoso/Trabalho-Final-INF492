"""
Architecture copied from:
https://towardsdatascience.com/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8
"""

from typing import Literal

import torch
from hparams import HParams
from torch import nn


class UnetBlock(nn.Module):
  def __init__(self, nf, ni, submodule=None, in_channels=None, dropout=False,
               innermost=False, outermost=False):
    super().__init__()
    self.outermost = outermost

    if in_channels is None: in_channels = nf

    downconv = nn.Conv2d(in_channels, ni, kernel_size=3,
                         stride=2, padding=1, bias=False)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = nn.BatchNorm2d(ni)
    uprelu = nn.ReLU(True)
    upnorm = nn.BatchNorm2d(nf)

    if outermost:
      upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=3,
                                  stride=2, padding=1)
      down = [downconv]
      up = [uprelu, upconv, nn.Tanh()]
      model = down + [submodule] + up
    elif innermost:
      upconv = nn.ConvTranspose2d(ni, nf, kernel_size=3,
                                  stride=2, padding=1, bias=False)
      down = [downrelu, downconv]
      up = [uprelu, upconv, upnorm]
      model = down + up
    else:
      upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=3,
                                  stride=2, padding=1, bias=False)
      down = [downrelu, downconv, downnorm]
      up = [uprelu, upconv, upnorm]
      if dropout: up += [nn.Dropout(0.5)]
      model = down + [submodule] + up

    self.model = nn.Sequential(*model)

  def forward(self, x):
    if self.outermost:
      return self.model(x)
    else:
      return torch.cat([x, self.model(x)], 1)


class Unet(nn.Module):
  def __init__(self):
    super().__init__()

    unet_block = UnetBlock(HParams.num_filters * 8, HParams.num_filters * 8, innermost=True)
    for _ in range(HParams.generator_n_down - 5):
      unet_block = UnetBlock(HParams.num_filters * 8, HParams.num_filters * 8, submodule=unet_block, dropout=True)
    out_filters = HParams.num_filters * 8
    for _ in range(3):
      unet_block = UnetBlock(out_filters // 2, out_filters, submodule=unet_block)
      out_filters //= 2
    self.model = UnetBlock(HParams.output_c, out_filters, in_channels=HParams.input_c, submodule=unet_block, outermost=True)

  def forward(self, x):
    return self.model(x)


class PatchDiscriminator(nn.Module):
  def __init__(self):
    super().__init__()

    model = [self.get_layers(HParams.input_c, HParams.num_filters, batch_norm=False)]
    model += [self.get_layers(HParams.num_filters * 2 ** i, HParams.num_filters * 2 ** (i + 1), stride=1 if i == (HParams.discriminator_n_down - 1) else 2) for i in range(HParams.discriminator_n_down)]
    model += [self.get_layers(HParams.num_filters * 2 ** HParams.discriminator_n_down, 1, stride=1, batch_norm=False, activation=False)]

    self.model = nn.Sequential(*model)

  def get_layers(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True, activation=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not batch_norm)]
    if batch_norm: layers += [nn.BatchNorm2d(out_channels)]
    if activation: layers += [nn.LeakyReLU(0.2, True)]
    return nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)



class GANLoss(nn.Module):
  def __init__(self, gan_mode: Literal['vanilla', 'lsgan'] = 'vanilla', real_label=1.0, fake_label=0.0):
    super().__init__()
    self.register_buffer('real_label', torch.tensor(real_label))
    self.register_buffer('fake_label', torch.tensor(fake_label))

    if gan_mode == 'vanilla':
      self.loss = nn.BCEWithLogitsLoss()
    elif gan_mode == 'lsgan':
      self.loss = nn.MSELoss()

  def get_labels(self, preds, target_is_real):
    if target_is_real:
      labels = self.real_label
    else:
      labels = self.fake_label
    return labels.expand_as(preds)

  def __call__(self, preds, target_is_real):
    labels = self.get_labels(preds, target_is_real)
    loss = self.loss(preds, labels)
    return loss
