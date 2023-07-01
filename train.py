from atexit import register
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
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

hparams = HParams()

generator = UnetGenerator(
    hparams.generator_input_channels,
    hparams.generator_output_channels,
    hparams.generator_num_down,
    hparams.generator_num_filters
).to(hparams.device)
optimizer_generator = torch.optim.AdamW(generator.parameters(), lr=hparams.lr_generator, betas=hparams.betas)

discriminator = NLayerDiscriminator(
    hparams.discriminator_input_channels,
    hparams.discriminator_num_filters_last_layer,
    hparams.discriminator_num_layers
).to(hparams.device)
optimizer_discriminator = torch.optim.AdamW(discriminator.parameters(), lr=hparams.lr_discriminator, betas=hparams.betas)

GANCriterion = GANLoss().to(hparams.device)
L1Criterion = torch.nn.L1Loss()

ROOT_DIR = Path(__file__).parent

tensorboard_root = ROOT_DIR / 'tensorboard'
models_root = ROOT_DIR / 'models'
generated_dir = ROOT_DIR / 'generated' / f'{hparams.dataset}'
writer = SummaryWriter(logdir=f'{tensorboard_root}/{hparams.dataset}/run - {len(list(tensorboard_root.glob("*")))}')

save_dir = generated_dir / f'run - {len(list(generated_dir.glob("*")))}'

save_dir.mkdir(parents=True, exist_ok=True)

save_every = int(len(train_loader) * 0.1)

def lab2rgb(L, AB, return_tensor=True):
    L = (L + 1.0) * 50.0
    AB = AB * 110.0
    lab = torch.cat([L, AB], dim=1)
    lab = lab[0].data.cpu().float().numpy()
    lab = np.transpose(lab.astype(np.float64), (1, 2, 0))
    rgb = color.lab2rgb(lab)
    if not return_tensor:
        return rgb
    return torch.from_numpy(rgb.transpose((2, 0, 1))).float()

def convert2rgb_and_return_grid(real_A, real_B, fake_B):
    rgb_real = lab2rgb(real_A, real_B)

    rgb_fake = lab2rgb(real_A, fake_B)

    return make_grid([rgb_real, rgb_fake], 2)


def save_image(real_A, fake_B, step):
    img = lab2rgb(real_A, fake_B, return_tensor=False)

    plt.title(f'Step {step}')
    plt.imsave(f'{save_dir}/{step:02d}.jpg', (img * 255).astype(np.uint8))


@torch.no_grad()
def estimate_loss():
    generator.eval()
    discriminator.eval()

    generator_loss = torch.zeros(len(val_loader))
    discriminator_loss = torch.zeros(len(val_loader))

    for eval_step, data in enumerate(tqdm(val_loader, leave=False, desc='Evaluation')):
        real_A = data['A'].to(hparams.device)
        real_B = data['B'].to(hparams.device)

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
        loss_generator_l1 = L1Criterion(fake_B, real_B) * hparams.lambda_L1
        generator_loss[eval_step] = (loss_generator_gan + loss_generator_l1).item()

    generator.train()
    discriminator.train()

    return generator_loss.mean().item(), discriminator_loss.mean().item()


def toggle_discriminator_require_grads():
    for param in discriminator.parameters():
        param.requires_grad = not param.requires_grad


best_loss_generator = float('inf')
best_loss_discriminator = float('inf')

step = 0

@register
def cleanup():
    if best_loss_discriminator == float('inf') or best_loss_generator == float('inf'):
        return
    writer.add_hparams(asdict(hparams), {'hparams/loss_generator': best_loss_generator, 'hparams/loss_discriminator': best_loss_discriminator}, global_step=step)

for epoch in range(hparams.epochs):
    for i, data in enumerate(tqdm(train_loader, leave=False, desc=f'Epoch {epoch}/{hparams.epochs}')):

        real_A = data['A'].to(hparams.device)
        real_B = data['B'].to(hparams.device)

        fake_B = generator(real_A)

        optimizer_discriminator.zero_grad(set_to_none=True)

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

        optimizer_generator.zero_grad(set_to_none=True)

        toggle_discriminator_require_grads()

        fake_AB = torch.cat([real_A, fake_B], dim=1)
        fake_pred = discriminator(fake_AB)

        loss_generator_gan = GANCriterion(fake_pred, True)
        loss_generator_l1 = L1Criterion(fake_B, real_B) * hparams.lambda_L1
        loss_generator = loss_generator_gan + loss_generator_l1
        loss_generator.backward()
        optimizer_generator.step()

        toggle_discriminator_require_grads()

        writer.add_scalar('train/discriminator', loss_discriminator.item(), step)

        writer.add_scalar('train/generator', loss_generator.item(), step)

        if step % hparams.visualization_interval == 0:
            writer.add_image('images', convert2rgb_and_return_grid(real_A, real_B, fake_B), step)

        if step % save_every == 0:
            save_image(real_A, fake_B, step)
        step += 1

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
        }, f'{models_root / f"{hparams.dataset}_generator.pth"}')

    if loss_discriminator < best_loss_discriminator:
        best_loss_discriminator = loss_discriminator
        torch.save({
            'epoch': epoch,
            'model': discriminator.state_dict(),
            'optimizer': optimizer_discriminator.state_dict()
        }, f'{models_root / f"{hparams.dataset}_discriminator.pth"}')
