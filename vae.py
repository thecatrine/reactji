import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid

from vae_modules import VectorQuantizedVAE, to_scalar

from torch.utils.tensorboard import SummaryWriter


# Edits
from loaders import datasets

device = torch.device('cuda')

writer = SummaryWriter()

def train(data_loader, model, optimizer, beta, steps, writer):
    for _, images, _ in data_loader:
        images = images.to(device)

        optimizer.zero_grad()
        x_tilde, z_e_x, z_q_x = model(images)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, images)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss = loss_recons + loss_vq + beta * loss_commit
        loss.backward()

        # Logs
        writer.add_scalar('loss/train/reconstruction', loss_recons.item(), steps)
        writer.add_scalar('loss/train/quantization', loss_vq.item(), steps)

        optimizer.step()
        steps += 1

    return steps

def test(data_loader, model, steps, writer):
    with torch.no_grad():
        loss_recons, loss_vq = 0., 0.
        for _, images, _ in data_loader:
            images = images.to(device)
            x_tilde, z_e_x, z_q_x = model(images)
            loss_recons += F.mse_loss(x_tilde, images)
            loss_vq += F.mse_loss(z_q_x, z_e_x)

        loss_recons /= len(data_loader)
        loss_vq /= len(data_loader)

    # Logs
    writer.add_scalar('loss/test/reconstruction', loss_recons.item(), steps)
    writer.add_scalar('loss/test/quantization', loss_vq.item(), steps)

    return loss_recons.item(), loss_vq.item()

def generate_samples(images, model):
    with torch.no_grad():
        images = images.to(device)
        x_tilde, _, _ = model(images)
    return x_tilde


data = datasets.NewTwitchDataset(batch_size=256, max_ts=1)


hidden_size = 256
k = 512
batch_size = 128
num_epochs = 10000
lr = 2e-4
beta = 1.0

model = VectorQuantizedVAE(3, hidden_size, k).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='VQ-VAE')\
    

    steps = 0
    best_loss = -1.
    for epoch in range(num_epochs):
        dataloaders = data.dataloaders()
        print(f"Epoch {epoch}")
        for dataloader in dataloaders:
            train_loader = dataloader['train']
            test_loader = dataloader['test']

            # Fixed images for Tensorboard
            _, fixed_images, _ = next(iter(test_loader))
            fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
            writer.add_image('original', fixed_grid, 0)

            
            steps = train(train_loader, model, optimizer, beta, steps, writer)
            loss, _ = test(test_loader, model, steps, writer)

            reconstruction = generate_samples(fixed_images, model)
            grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
            writer.add_image('reconstruction', grid, epoch + 1)

            if (epoch == 0) or (loss < best_loss):
                best_loss = loss
                with open('vqvae/best.pt', 'wb') as f:
                    torch.save(model.state_dict(), f)

        with open(f'vqvae/model_{epoch + 1}.pt', 'wb') as f:
            torch.save(model.state_dict(), f)
