from datetime import datetime
from pathlib import Path

import torch
from dataset import make_dataloaders
from hparams import HParams
from models import GANLoss, PatchDiscriminator, Unet
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

train_loader, val_loader = make_dataloaders()

G = Unet().to(HParams.device)
optim_G = torch.optim.AdamW(G.parameters(), lr=HParams.lr_g, betas=HParams.betas)

D = PatchDiscriminator().to(HParams.device)
optim_D = torch.optim.AdamW(D.parameters(), lr=HParams.lr_d, betas=HParams.betas)

GANCriterion = GANLoss().to(HParams.device)
L1Criterion = torch.nn.L1Loss()

tensorboard_root = Path(__file__).parent / 'tensorboard'
models_root = Path(__file__).parent / 'models'
writer = SummaryWriter(logdir=f'{tensorboard_root} / {datetime.now()}')


@torch.no_grad()
def estimate_loss(step: int):
  G.eval()
  D.eval()

  g_loss = torch.zeros(len(val_loader))
  d_loss = torch.zeros(len(val_loader))

  for eval_step, batch in enumerate(tqdm(val_loader, leave=False, desc='Evaluation')):
    L = batch['L'].to(HParams.device)
    AB = batch['ab'].to(HParams.device)

    fake_color = G(L)

    fake_img = torch.cat([L, fake_color], dim=1)
    fake_preds = D(fake_img.detach())

    loss_d_fake = GANCriterion(fake_preds, False)

    real_img = torch.cat([L, AB], dim=1)
    real_preds = D(real_img)
    loss_d_real = GANCriterion(real_preds, True)

    d_loss[eval_step] = ((loss_d_fake + loss_d_real) / 2).item()

    loss_g_gan = GANCriterion(fake_preds, True)
    loss_g_l1 = L1Criterion(fake_color, AB) * HParams.lambda_L1

    g_loss[eval_step] = (loss_g_gan + loss_g_l1).item()

  writer.add_image('images', make_grid([real_img, fake_img], 2), step)

  G.train()
  D.train()

  return g_loss.mean().item(), d_loss.mean().item()


best_loss_g = float('inf')
best_loss_d = float('inf')


for epoch in range(HParams.epochs):
  for i, batch in enumerate(tqdm(train_loader, leave=False, desc=f'Epoch {epoch}/{HParams.epochs}')):

    step = epoch * len(train_loader) + i

    L = batch['L'].to(HParams.device)
    AB = batch['ab'].to(HParams.device)

    fake_color = G(L)

    optim_G.zero_grad(True)

    fake_img = torch.cat([L, fake_color], dim=1)
    fake_preds = D(fake_img.detach())
    loss_d_fake = GANCriterion(fake_preds, False)

    real_img = torch.cat([L, AB], dim=1)
    real_preds = D(real_img)
    loss_d_real = GANCriterion(real_preds, True)

    loss_d = (loss_d_fake + loss_d_real) / 2

    writer.add_scalar('train/discriminator', loss_d.item(), step)

    loss_d.backward()
    optim_D.step()

    loss_g_gan = GANCriterion(fake_preds, True)
    loss_g_l1 = L1Criterion(fake_color, AB) * HParams.lambda_L1

    loss_g = loss_g_gan + loss_g_l1

    writer.add_scalar('train/generator', loss_g.item(), step)

    loss_g.backward()
    optim_G.step()

  loss_g, loss_d = estimate_loss(step)
  writer.add_scalar('val/generator', loss_g, step)
  writer.add_scalar('val/discriminator', loss_d, step)

  for name, weight in G.named_parameters():
    writer.add_histogram(f'weights/generator/{name}', weight.clone().cpu().data.numpy(), step)
  for name, weight in D.named_parameters():
    writer.add_histogram(f'weights/discriminator/{name}', weight.clone().cpu().data.numpy(), step)

  if loss_g < best_loss_g:
    best_loss_g = loss_g
    torch.save({
      'epoch': epoch,
      'model': G.state_dict(),
      'optimizer': optim_G.state_dict()
    }, f'{models_root / "generator.pth"}')

  if loss_d < best_loss_d:
    best_loss_d = loss_d
    torch.save({
      'epoch': epoch,
      'model': D.state_dict(),
      'optimizer': optim_D.state_dict()
    }, f'{models_root / "discriminator.pth"}')