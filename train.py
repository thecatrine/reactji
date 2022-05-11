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

LR = float(os.environ.get('LR', '0.001'))
log.info(f'Using LR={LR}')
BATCH_SZ = int(os.environ.get('BATCH_SZ', '128'))
log.info(f'Using BATCH_SZ={BATCH_SZ}')

random.seed(0)
np.random.seed(0)

p = argparse.ArgumentParser()
p.add_argument('--resume', default="")
args = p.parse_args()

# Data
log.info('Loading twitch dataset...')
data = datasets.ImagenetData(batch_size=BATCH_SZ)

# Model
log.info('Constructing model...')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {device}")
model = diffuser.Diffuser(dropout_rate=0.1)
if args.resume:
    log.info("Resuming from {}...".format(args.resume))
    model.load_state_dict(torch.load(args.resume))
else:
    log.info('Initializing...')
log.info('Sending model to device...')
model.to(device)
# optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)

# Tensorboard
log.info('Configuring tensorboard...')
id_loss = 3e-3
writer = SummaryWriter()

def train_one_epoch(train_data, cur_epoch):
    model.train(True)
    running_loss = 0.
    last_loss = 0.

    running_loss = 0
    start_time = time.time()
    processed = 0
    for i, batches in enumerate(train_data):
        timesteps, inputs, expected_outputs = [x.to(device) for x in batches]
        processed += inputs.shape[0]

        optimizer.zero_grad()
        outputs = model(inputs, timesteps)
        loss = loss_fn(outputs, expected_outputs) / id_loss
        scale = min(i/(1+i), 0.99)
        running_loss = scale*running_loss + (1-scale)*loss.item()
        loss.backward()
        optimizer.step()

        writer.add_scalar('train_loss_scaled', loss,
                          cur_epoch + (i*BATCH_SZ)/len(train_data))
        if (i+1)%20 == 0:
            log.info(f'Epoch {cur_epoch:03} | Train Batch {i+1:06} | Loss {running_loss:.10f} | ex/s {processed / (time.time() - start_time):03.0f}')

def scaled_test_loss(test_data):
    model.train(False)
    with torch.no_grad():
        running_vloss = 0.
        test_len = 0
        for vdata in test_data:
            s, vinputs, vlabels = vdata
            s, vinputs, vlabels = s.to(device), vinputs.to(device), vlabels.to(device)
            voutputs = model(vinputs, s)
            vloss = loss_fn(voutputs, vlabels) / id_loss
            running_vloss += vloss.item()
            test_len += 1
        return running_vloss / test_len

EPOCHS = 1000
best_test_loss = 10e10

log.info('Training...')
for epoch in range(EPOCHS):
    log.info(f'Epoch {epoch:03}:')

    dataloaders = data.dataloaders()
    train_data = dataloaders['train']
    test_data = dataloaders['test']

    train_loss = train_one_epoch(train_data, epoch)
    log.info('Testing...')
    test_loss = scaled_test_loss(test_data)
    log.info(f'Epoch {epoch:03} | Test | {test_loss:.10f}')
    writer.add_scalar('test_loss_scaled', test_loss)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), f"best_model_{epoch}.pth")
    torch.save(model.state_dict(), f"cur_model.pth")
