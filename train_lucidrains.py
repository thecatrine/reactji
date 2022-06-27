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

BATCH_SZ = int(os.environ.get('BATCH_SZ', '128'))
log.info(f'Using BATCH_SZ={BATCH_SZ}')


EPOCHS = 10000
epoch = 0
best_test_loss = 10e10


log.info('Constructing unet...')
unet = Unet(
    dim=128,
    dim_mults=(1, 2, 3),
).cuda()

decoder = Decoder(
    unet=unet,
    image_size=28,
    timesteps=1000,
    unconditional=True,
).cuda()

decoder_trainer = DecoderTrainer(decoder)

# Data
log.info('Loading twitch dataset...')
data = datasets.NewTwitchDataset(batch_size=BATCH_SZ, manual_shuffle=True)

log.info('Training...')
decoder_trainer.load('chk_lucidrains.pth')
while epoch < EPOCHS:
    dataloaders = data.dataloaders()
    train_data = dataloaders['train']
    test_data = dataloaders['test']

    tensors = train_data.dataset.tensors.cuda()
    chunk_sz = BATCH_SZ*1024
    for chunk_start in range(0, len(tensors), chunk_sz):
        loss = decoder_trainer(tensors[chunk_start:chunk_start+chunk_sz],
                               max_batch_size=BATCH_SZ)
        log.info(f'Train Loss: {loss}')

        if True: #(chunk_start // chunk_sz) % 10 == 0:
            log.info('Saving...')
            decoder_trainer.save('chk_lucidrains.pth')
