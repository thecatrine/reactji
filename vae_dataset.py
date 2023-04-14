from multiprocessing import dummy

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from loaders import datasets
from models import diffuser
from loaders import loader_utils
import random
import zipfile
import os
import logging
import argparse

from PIL import Image


from vae_modules import VectorQuantizedVAE, to_scalar

hidden_size = 512
k = 2024
batch_size = 128
num_epochs = 10000
lr = 2e-4
beta = 1.0

device = torch.device('cuda')

data = datasets.NewTwitchDataset(batch_size=256, max_ts=1)
dataloaders = data.dataloaders()

model = VectorQuantizedVAE(3, hidden_size, k).to(device)
model.load_state_dict(torch.load('./model_1533.pt'))
model.eval()

batches_of_tokens = []

i = 0
for dataloader in dataloaders:
    print(f"Loader {i}")
    i += 1
    train_loader = dataloader['train']
    test_loader = dataloader['test']

    for _, inputs, _ in train_loader:
        model_outputs, encoder_cont, encoder_q = model(inputs.to(device))

        tokens = model.codebook(encoder_cont)
        tokens = tokens.reshape(tokens.shape[0], -1)

        batches_of_tokens.append(tokens.cpu().detach())

tokens_tensor = torch.cat(batches_of_tokens, dim=0)
torch.save(tokens_tensor, "vae2/tokens_train.pt")
        
print(tokens_tensor.shape)
