from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from skimage import color
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from dataset import make_dataloaders
from hparams import HParams
from models import GANLoss, NLayerDiscriminator, UnetGenerator

train_loader, val_loader = make_dataloaders()

generator = UnetGenerator(
    HParams.generator_input_channels,
    HParams.generator_output_channels,
    HParams.generator_num_down,
    HParams.generator_num_filters
).to(HParams.device)
optimizer_generator = torch.optim.AdamW(generator.parameters(), lr=HParams.lr_generator, betas=HParams.betas)

discriminator = NLayerDiscriminator(
    HParams.discriminator_input_channels,
    HParams.discriminator_num_filters_last_layer,
    HParams.discriminator_num_layers
).to(HParams.device)
optimizer_discriminator = torch.optim.AdamW(discriminator.parameters(), lr=HParams.lr_discriminator, betas=HParams.betas)

GANCriterion = GANLoss().to(HParams.device)
L1Criterion = torch.nn.L1Loss()

tensorboard_root = Path(__file__).parent / 'tensorboard'
models_root = Path(__file__).parent / 'models'
writer = SummaryWriter(logdir=f'{tensorboard_root}/{datetime.now()}')

def convert2rgb_and_return_grid(real_A, real_B, fake_B):
    real_B2 = real_B * 110.0
    real_A2 = (real_A + 1.0) * 50.0
    fake_B2 = fake_B * 110.0

    lab_real = torch.cat([real_A2, real_B2], dim=1)
    lab_real = lab_real[0].data.cpu().float().numpy()
    lab_real = np.transpose(lab_real.astype(np.float64), (1, 2, 0))
    rgb_real = color.lab2rgb(lab_real)
    rgb_real = torch.from_numpy(rgb_real.transpose((2, 0, 1))).float()

    lab_fake = torch.cat([real_A2, fake_B2], dim=1)
    lab_fake = lab_fake[0].data.cpu().float().numpy()
    lab_fake = np.transpose(lab_fake.astype(np.float64), (1, 2, 0))
    rgb_fake = color.lab2rgb(lab_fake)
    rgb_fake = torch.from_numpy(rgb_fake.transpose((2, 0, 1))).float()

    return make_grid([rgb_real, rgb_fake], 2)


@torch.no_grad()
def estimate_loss():
    generator.eval()
    discriminator.eval()

    generator_loss = torch.zeros(len(val_loader))
    discriminator_loss = torch.zeros(len(val_loader))

    for eval_step, data in enumerate(tqdm(val_loader, leave=False, desc='Evaluation')):
        real_A = data['A'].to(HParams.device)
        real_B = data['B'].to(HParams.device)

        fake_B = generator(real_A)

        fake_AB = torch.cat([real_A, fake_B], dim=1)
        fake_pred = discriminator(fake_AB)
        loss_discriminator_fake = GANCriterion(fake_pred, False)
        real_AB = torch.cat([real_A, real_B], dim=1)
        real_pred = discriminator(real_AB)
        loss_discriminator_real = GANCriterion(real_pred, True)
        loss_discriminator = (loss_discriminator_fake + loss_discriminator_real) * 0.5
        discriminator_loss[eval_step] = loss_discriminator.item()

        fake_AB = torch.cat([real_A, fake_B], dim=1)
        fake_pred = discriminator(fake_AB)
        loss_generator_gan = GANCriterion(fake_pred, True)
        loss_generator_l1 = L1Criterion(fake_B, real_B) * HParams.lambda_L1
        generator_loss[eval_step] = (loss_generator_gan + loss_generator_l1).item()

    generator.train()
    discriminator.train()

    return generator_loss.mean().item(), discriminator_loss.mean().item()


best_loss_generator = float('inf')
best_loss_discriminator = float('inf')


for epoch in range(HParams.epochs):
    for i, data in enumerate(tqdm(train_loader, leave=False, desc=f'Epoch {epoch}/{HParams.epochs}')):

        step = epoch * len(train_loader) + i

        real_A = data['A'].to(HParams.device)
        real_B = data['B'].to(HParams.device)

        fake_B = generator(real_A)

        optimizer_discriminator.zero_grad()

        fake_AB = torch.cat([real_A, fake_B], dim=1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        fake_pred = discriminator(fake_AB.detach())
        loss_discriminator_fake = GANCriterion(fake_pred, False)

        real_AB = torch.cat([real_A, real_B], dim=1)
        real_pred = discriminator(real_AB)
        loss_discriminator_real = GANCriterion(real_pred, True)
        # combine loss and calculate gradients
        loss_discriminator = (loss_discriminator_fake + loss_discriminator_real) * 0.5
        loss_discriminator.backward()
        optimizer_discriminator.step()

        optimizer_generator.zero_grad()

        fake_AB = torch.cat([real_A, fake_B], dim=1)
        fake_pred = discriminator(fake_AB)

        loss_generator_gan = GANCriterion(fake_pred, True)
        loss_generator_l1 = L1Criterion(fake_B, real_B) * HParams.lambda_L1
        loss_generator = loss_generator_gan + loss_generator_l1
        loss_generator.backward()
        optimizer_generator.step()

        writer.add_scalar('train/discriminator', loss_discriminator.item(), step)

        writer.add_scalar('train/generator', loss_generator.item(), step)

        if step % HParams.visualization_interval == 0:
            writer.add_image('images', convert2rgb_and_return_grid(real_A, real_B, fake_B), step)

    loss_generator, loss_discriminator = estimate_loss()
    writer.add_scalar('val/generator', loss_generator, step)
    writer.add_scalar('val/discriminator', loss_discriminator, step)

    for name, weight in generator.named_parameters():
        writer.add_histogram(f'weights/generator/{name}', weight.clone().cpu().data.numpy(), step)
    for name, weight in discriminator.named_parameters():
        writer.add_histogram(f'weights/discriminator/{name}', weight.clone().cpu().data.numpy(), step)

    if loss_generator < best_loss_generator:
        best_loss_generator = loss_generator
        torch.save({
            'epoch': epoch,
            'model': generator.state_dict(),
            'optimizer': optimizer_generator.state_dict()
        }, f'{models_root / "generator.pth"}')

    if loss_discriminator < best_loss_discriminator:
        best_loss_discriminator = loss_discriminator
        torch.save({
            'epoch': epoch,
            'model': discriminator.state_dict(),
            'optimizer': optimizer_discriminator.state_dict()
        }, f'{models_root / "discriminator.pth"}')
