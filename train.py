from loaders import datasets
from models import diffuser
from models import glide_model
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

LR = float(os.environ.get('LR', '3e-5'))
log.info(f'Using LR={LR}')
BATCH_SZ = int(os.environ.get('BATCH_SZ', '256'))
log.info(f'Using BATCH_SZ={BATCH_SZ}')

TRAIN_ON_NOISE = os.environ.get('TRAIN_ON_NOISE', '1')
TRAIN_ON_NOISE = bool(int(TRAIN_ON_NOISE))
log.info(f'Using TRAIN_ON_NOISE={TRAIN_ON_NOISE}')

PRECISION = os.environ.get('PRECISION', 'AUTO')
assert PRECISION in ['AUTO', '32', '16']
USE_AUTOCAST = (PRECISION == 'AUTO')
log.info(f'Using PRECISION={PRECISION}')

FORCE_WARMUP = os.environ.get('FORCE_WARMUP', '0')
FORCE_WARMUP = bool(int(FORCE_WARMUP))
log.info(f'Using FORCE_WARMUP={FORCE_WARMUP}')

EPOCH_LR_SCALING = os.environ.get('EPOCH_LR_SCALING', '1')
EPOCH_LR_SCALING = bool(int(EPOCH_LR_SCALING))
log.info(f'Using EPOCH_LR_SCALING={EPOCH_LR_SCALING}')

LOSS_SCALING = os.environ.get('LOSS_SCALING', '1')
LOSS_SCALING = bool(int(LOSS_SCALING))
log.info(f'Using LOSS_SCALING={LOSS_SCALING}')

MANUAL_SHUFFLE = os.environ.get('MANUAL_SHUFFLE', '1')
MANUAL_SHUFFLE = bool(int(MANUAL_SHUFFLE))
log.info(f'Using MANUAL_SHUFFLE={MANUAL_SHUFFLE}')

LOSS_FN = os.environ.get('LOSS_FN', 'L2')
assert LOSS_FN in ['L2', 'SMOOTH_L1']
log.info(f'Using LOSS_FN={LOSS_FN}')

INPUT_DATASET = os.environ.get('INPUT_DATASET', '28')
assert INPUT_DATASET in ['28', '112']
log.info(f'Using INPUT_DATASET={INPUT_DATASET}')
if INPUT_DATASET == '28':
    diffuser_opts = {
        'normalization_groups': 32,
        'channels': 256,
        'num_head_channels': 64,
        'num_residuals': 6,
        'channel_multiple_schedule': [1, 2, 3],
    }
elif INPUT_DATASET == '112':
    diffuser_opts = {
        'normalization_groups': 2,
        'channels': 12,
        'num_head_channels': 4,
        'num_residuals': 3,
        'channel_multiple_schedule': [1, 2, 3, 6, 12],
    }

CONDITION_ON_DOWNSAMPLE = os.environ.get('CONDITION_ON_DOWNSAMPLE', '')
if CONDITION_ON_DOWNSAMPLE == '':
    CONDITION_ON_DOWNSAMPLE = None
    diffuser_opts['in_channels'] = 3
else:
    CONDITION_ON_DOWNSAMPLE = int(CONDITION_ON_DOWNSAMPLE)
    assert CONDITION_ON_DOWNSAMPLE == 4
    diffuser_opts['in_channels'] = 6
log.info(f'Using CONDITION_ON_DOWNSAMPLE={CONDITION_ON_DOWNSAMPLE}')

RUN_NAME = os.environ.get('RUN_NAME', '')
assert RUN_NAME != ''
log.info(f'Using RUN_NAME={RUN_NAME}')

random.seed(0)
np.random.seed(0)

p = argparse.ArgumentParser()
p.add_argument('--resume', default="")

p.add_argument('--glide', action='store_true')
args = p.parse_args()

EPOCHS = 10000
epoch = 0
best_test_loss = 10e10

prefix_path = f'run_{RUN_NAME}'
tensorboard_path = f'runs/{prefix_path}'
if not os.path.exists(prefix_path):
    os.mkdir(prefix_path)
if not os.path.exists(tensorboard_path):
    os.mkdir(tensorboard_path)

def save_all(_path):
    path = f'{prefix_path}/{_path}'
    log.info(f'Saving to {path}...')
    global epoch, best_test_loss, model, optimizer, lr_scheduler
    torch.save({
        'epoch': epoch,
        'best_test_loss': best_test_loss,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }, path)

def load_all(path):
    global epoch, best_test_loss, model, optimizer, lr_scheduler
    loaded = torch.load(path)
    if not isinstance(loaded, dict) or 'model' not in loaded:
        loaded = {
            'epoch': 0,
            'best_test_loss': 10e10,
            'model': loaded,
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
        }
    epoch = loaded['epoch']
    best_test_loss = loaded['best_test_loss']
    model.load_state_dict(loaded['model'])
    # try doing this
    if PRECISION == '16':
        model = model.to(torch.float16)
    #import pdb; pdb.set_trace()

    if not FORCE_WARMUP:
        optimizer.load_state_dict(loaded['optimizer'])
        lr_scheduler.load_state_dict(loaded['lr_scheduler'])

# Data
log.info('Loading twitch dataset...')
if INPUT_DATASET == '28':
    data = datasets.NewTwitchDataset(batch_size=BATCH_SZ, manual_shuffle=MANUAL_SHUFFLE)
else:
    data = datasets.BigTwitchDataset(batch_size=BATCH_SZ, manual_shuffle=MANUAL_SHUFFLE)

# Model
log.info('Constructing model...')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {device}")

if args.glide:
    log.info("Using model from glide repo")
    model = glide_model.create_model()
else:
    model = diffuser.Diffuser(**diffuser_opts)

if PRECISION == '16':
    model = model.to(torch.float16)

log.info('Sending model to device...')
model = model.to(device)
old_loss_fn = torch.nn.MSELoss()
old_id_loss = 3e-3
if LOSS_FN == 'L2':
    loss_fn = torch.nn.MSELoss()
elif LOSS_FN == 'SMOOTH_L1':
    loss_fn = torch.nn.SmoothL1Loss(beta=0.1)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR,
                              weight_decay=1e-3, betas=(0.9, 0.999))

START_FACTOR = 1
if FORCE_WARMUP:
    START_FACTOR=1e-6
lr_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=START_FACTOR,
    end_factor=1.0,
    total_iters=9000,
)
lr_epoch_scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=0.95,
)

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
writer = SummaryWriter(log_dir=tensorboard_path)

def gradient_norm():
    parameters = [p for p in model.parameters() if p.grad is not None]
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]), 2)
    return total_norm

def max_gradient():
    parameters = [p for p in model.parameters() if p.grad is not None]
    return torch.max(torch.stack([torch.max(p.grad.detach()) for p in parameters]))

def train_one_sub_epoch(train_data):
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
        true_target = expected_outputs
        if TRAIN_ON_NOISE:
            expected_outputs = inputs - expected_outputs
        if inputs.shape[0] < BATCH_SZ/2:
            log.warning('SKIPPING SMALL BATCH')
            continue
        processed += inputs.shape[0]

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=USE_AUTOCAST):
            if PRECISION != '32':
                inputs = inputs.to(torch.float16)
                timesteps = timesteps.to(torch.float16)
            if CONDITION_ON_DOWNSAMPLE is not None:
                scale = CONDITION_ON_DOWNSAMPLE
                downsampled = F.avg_pool_2d(true_target, kernel_size=scale, stride=scale)
                blurred = F.gaussian_blur(downsampled, kernel_size=3)
                upsampled = F.interpolate(blurred, scale_factor=scale, mode='nearest')
                inputs = torch.cat(inputs, upsampled, dim=1)
            outputs = model(inputs, timesteps)
            # print('space', torch.cuda.memory_allocated(0))
            id_loss = loss_fn(inputs, expected_outputs)
            if LOSS_SCALING:
                true_loss = torch.mean(torch.square(outputs - expected_outputs) /
                                       torch.sqrt(timesteps + 1).reshape(-1, 1, 1, 1))
            else:
                true_loss = loss_fn(outputs, expected_outputs)

        loss = true_loss / id_loss
        with torch.no_grad():
            old_loss = old_loss_fn(outputs, expected_outputs) / old_id_loss

        scale = min(i/(1+i), 0.99)
        running_loss = scale*running_loss + (1-scale)*loss.item()
        running_old_loss = scale*running_old_loss + (1-scale)*old_loss.item()

        if USE_AUTOCAST:
            scaler.scale(true_loss).backward()
            scaler.unscale_(optimizer)
        else:
            true_loss.backward()

        grad_norm = gradient_norm()
        grad_max = max_gradient()
        # log.info(f'gradient norm: {grad_norm:.3f} max: {grad_max:.3f} loss: {true_loss}')
        # TODO: Are these numbers good?
        if LOSS_SCALING:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        if grad_norm > 5 or grad_max > 1:
            log.warning(f'HIGH GRADIENT {grad_norm} {grad_max} (post-clip: {gradient_norm()})')

        if USE_AUTOCAST:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        last_lr = lr_scheduler.get_last_lr()[0]
        lr_scheduler.step()

        cur_step = (epoch*len(train_data) + i)*BATCH_SZ

        writer.add_scalar('lr', last_lr)
        writer.add_scalar('train_loss_scaled', loss, cur_step)
        writer.add_scalar('train_old_loss_scaled', old_loss, cur_step)
        if (i+1)%20 == 0:
            log.info(f'E/{epoch:03} B/{i+1:06} | L {running_loss:.6f} | Old L {running_old_loss:.6f} | ex/s {processed / (time.time() - start_time):03.0f} | lr {last_lr:0.3g}')
            log.info(f'      {true_loss}')
        if (i+1)%1000 == 0:
            save_all(f"cur_model.pth")
    if EPOCH_LR_SCALING:
        lr_epoch_scheduler.step()

def scaled_test_loss(test_data):
    model.train(False)
    with torch.no_grad():
        running_vloss = 0.
        test_len = 0
        for vdata in test_data:
            s, inputs, expected_outputs = vdata
            if TRAIN_ON_NOISE:
                expected_outputs = inputs - expected_outputs
            s = s.to(torch.float32)
            s, inputs, expected_outputs = s.to(device), inputs.to(device), expected_outputs.to(device)
            voutputs = model(inputs, s)
            id_loss = loss_fn(inputs, expected_outputs)
            vloss = loss_fn(voutputs, expected_outputs) / id_loss
            running_vloss += vloss.item()
            test_len += 1
        return running_vloss / test_len

log.info('Training...')
while epoch < EPOCHS:
    log.info(f'Epoch {epoch:03}:')

    all_dataloaders = data.dataloaders()
    test_loss_acc = 0
    n_test_loss = 0
    for dataloaders in all_dataloaders:
        train_data = dataloaders['train']
        test_data = dataloaders['test']

        train_loss = train_one_sub_epoch(train_data)
        log.info('Testing...')
        test_loss = scaled_test_loss(test_data)
        test_loss_acc += test_loss
        n_test_loss += 1

    test_loss = test_loss_acc / n_test_loss
    log.info(f'Epoch {epoch:03} | Test | {test_loss:.10f}')
    writer.add_scalar('test_loss_scaled', test_loss)

    epoch += 1
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        save_all(f"best_model_{epoch}.pth")
    save_all(f"cur_model.pth")
