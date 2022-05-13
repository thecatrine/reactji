from loaders import datasets
from models import diffuser
from models import diffuser
from torch.utils.tensorboard import SummaryWriter
import logging
import argparse
import matplotlib.pyplot as plt
import random
import torch
import torchvision
import zipfile
import numpy as np
import os
import time

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)
log = logging.getLogger(__name__)

LR = float(os.environ.get('LR', '1e-4'))
log.info(f'Using LR={LR}')
BATCH_SZ = int(os.environ.get('BATCH_SZ', '256'))
log.info(f'Using BATCH_SZ={BATCH_SZ}')

random.seed(0)
np.random.seed(0)

p = argparse.ArgumentParser()
p.add_argument('--resume', default="")
args = p.parse_args()

EPOCHS = 10000
epoch = 0
best_test_loss = 10e10

def save_all(path):
    torch.save({
        'epoch': epoch,
        'best_test_loss': best_test_loss,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, path)

def load_all(path):
    global epoch, best_test_loss, model, optimizer
    loaded = torch.load(path)
    if not isinstance(loaded, dict) or 'model' not in loaded:
        loaded = {
            'epoch': 0,
            'best_test_loss': 10e10,
            'model': loaded,
            'optimizer': optimizer.state_dict(),
        }
    epoch = loaded['epoch']
    best_test_loss = loaded['best_test_loss']
    model.load_state_dict(loaded['model'])
    optimizer.load_state_dict(loaded['optimizer'])

# Data
log.info('Loading twitch dataset...')
data = datasets.ImagenetData(batch_size=BATCH_SZ)

# Model
log.info('Constructing model...')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {device}")
model = diffuser.Diffuser(dropout_rate=0.1)
# model = model.to(torch.float16)
log.info('Sending model to device...')
model = model.to(device)
old_loss_fn = torch.nn.MSELoss()
old_id_loss = 3e-3
loss_fn = torch.nn.L1Loss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR,
                              weight_decay=1e-3, betas=(0.9, 0.999))

if args.resume:
    log.info("Resuming from {}...".format(args.resume))
    load_all(args.resume)
else:
    log.info('Initializing...')

scaler = torch.cuda.amp.GradScaler()
print(torch.cuda.get_device_properties(0).total_memory)
print(torch.cuda.memory_reserved(0))
print(torch.cuda.memory_allocated(0))
print(sum(x.numel() for x in model.parameters()))

# Tensorboard
log.info('Configuring tensorboard...')
writer = SummaryWriter()

def gradient_norm():
    parameters = [p for p in model.parameters() if p.grad is not None]
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]), 2)
    return total_norm

def max_gradient():
    parameters = [p for p in model.parameters() if p.grad is not None]
    return torch.max(torch.stack([torch.max(p.grad.detach()) for p in parameters]))

def train_one_epoch(train_data):
    model.train(True)
    running_loss = 0.
    running_old_loss = 0.
    last_loss = 0.

    running_loss = 0
    start_time = time.time()
    processed = 0
    for i, batches in enumerate(train_data):
        timesteps, inputs, expected_outputs = [
            x.to(device) for x in batches
        ]
        if inputs.shape[0] < BATCH_SZ/2:
            log.warning('SKIPPING SMALL BATCH')
            continue
        processed += inputs.shape[0]

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs, timesteps)
            # print('space', torch.cuda.memory_allocated(0))
            id_loss = loss_fn(inputs, expected_outputs)
            true_loss = loss_fn(outputs, expected_outputs)

        loss = true_loss / id_loss
        with torch.no_grad():
            old_loss = old_loss_fn(outputs, expected_outputs) / old_id_loss

        scale = min(i/(1+i), 0.99)
        running_loss = scale*running_loss + (1-scale)*loss.item()
        running_old_loss = scale*running_old_loss + (1-scale)*old_loss.item()

        scaler.scale(true_loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = gradient_norm()
        grad_max = max_gradient()
        # log.info(f'gradient norm: {grad_norm:.3f} max: {grad_max:.3f} loss: {true_loss}')
        # TODO: Are these numbers good?
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        if grad_norm > 5 or grad_max > 1:
            log.warning(f'HIGH GRADIENT {grad_norm} {grad_max} (post-clip: {gradient_norm()})')
        scaler.step(optimizer)
        scaler.update()

        cur_step = (epoch*len(train_data) + i)*BATCH_SZ

        writer.add_scalar('train_loss_scaled', loss, cur_step)
        writer.add_scalar('train_old_loss_scaled', old_loss, cur_step)
        if (i+1)%20 == 0:
            log.info(f'E/{epoch:03} B/{i+1:06} | L {running_loss:.6f} | Old L {running_old_loss:.6f} | ex/s {processed / (time.time() - start_time):03.0f}')

def scaled_test_loss(test_data):
    model.train(False)
    with torch.no_grad():
        running_vloss = 0.
        test_len = 0
        for vdata in test_data:
            s, vinputs, vlabels = vdata
            s, vinputs, vlabels = s.to(device), vinputs.to(device), vlabels.to(device)
            voutputs = model(vinputs, s)
            id_loss = loss_fn(vinputs, vlabels)
            vloss = loss_fn(voutputs, vlabels) / id_loss
            running_vloss += vloss.item()
            test_len += 1
        return running_vloss / test_len

log.info('Training...')
while epoch < EPOCHS:
    log.info(f'Epoch {epoch:03}:')

    dataloaders = data.dataloaders()
    train_data = dataloaders['train']
    test_data = dataloaders['test']

    train_loss = train_one_epoch(train_data)
    log.info('Testing...')
    test_loss = scaled_test_loss(test_data)
    log.info(f'Epoch {epoch:03} | Test | {test_loss:.10f}')
    writer.add_scalar('test_loss_scaled', test_loss)

    epoch += 1
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        save_all(f"best_model_{epoch}.pth")
    save_all(f"cur_model.pth")
