import sys
sys.path.append('./DALLE2-pytorch')
from dalle2_pytorch import Unet, Decoder, DecoderTrainer

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

BATCH_SZ = int(os.environ.get('BATCH_SZ', '200'))
log.info(f'Using BATCH_SZ={BATCH_SZ}')

RUN_NAME = os.environ.get('RUN_NAME', '')
assert RUN_NAME != ''
log.info(f'Using RUN_NAME={RUN_NAME}')

prefix_path = f'run_{RUN_NAME}'
tensorboard_path = f'runs/{prefix_path}'
if not os.path.exists(prefix_path):
    os.mkdir(prefix_path)
if not os.path.exists(tensorboard_path):
    os.mkdir(tensorboard_path)

# Tensorboard
log.info('Configuring tensorboard...')
writer = SummaryWriter(log_dir=tensorboard_path)

EPOCHS = 10000
epoch = 0
best_test_loss = 10e10

log.info('Constructing unet...')
# unet = Unet(
#     dim=128,
#     dim_mults=(1, 2, 3),
# ).cuda()

unet = diffuser.Diffuser().cuda()

decoder = Decoder(
    unet=unet,
    image_size=28,
    timesteps=1000,
    unconditional=True,
    learned_variance=False,
).cuda()

decoder_trainer = DecoderTrainer(decoder)

# Data
log.info('Loading twitch dataset...')
data = datasets.NewTwitchDataset(batch_size=BATCH_SZ, manual_shuffle=True)
old_id_loss = 3e-3

log.info('Training...')
# decoder_trainer.load('chk_lucidrains.pth')
while epoch < EPOCHS:
    dataloaders = data.dataloaders()
    train_data = dataloaders['train']
    test_data = dataloaders['test']

    tensors = train_data.dataset.tensors.cuda()
    chunk_sz = BATCH_SZ
    for chunk_start in range(0, len(tensors), chunk_sz):
        loss = decoder_trainer(tensors[chunk_start:chunk_start+chunk_sz],
                               max_batch_size=BATCH_SZ)
        decoder_trainer.update()
        log.info(f'Train Loss: {loss}')
        writer.add_scalar('train_loss_lucidrains', loss,
                          chunk_start + epoch * len(tensors))
        writer.add_scalar('train_old_loss_scaled', loss / old_id_loss,
                          chunk_start + epoch * len(tensors))

        if (chunk_start // chunk_sz) % 1000 == 0:
            log.info('Saving...')
            decoder_trainer.save('chk_lucidrains.pth')
            # log.info('Saving test images...')
            # images = decoder_trainer.sample(batch_size=8, max_batch_size=8)
            # torch.save(images, 'lucidrains_images.pt')
