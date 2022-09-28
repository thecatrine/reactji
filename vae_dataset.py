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
from transformers import CLIPProcessor, CLIPModel

from vae_modules import VectorQuantizedVAE, to_scalar

hidden_size = 256
k = 512
batch_size = 128
num_epochs = 10000
lr = 2e-4
beta = 1.0

device = torch.device('cuda')

data = datasets.NewTwitchDataset(batch_size=256, max_ts=1)
dataloaders = data.dataloaders()

model = VectorQuantizedVAE(3, hidden_size, k).to(device)
model.load_state_dict(torch.load('vqvae/model_1159.pt'))
model.eval()

for dataloader in dataloaders:
    train_loader = dataloader['train']
    test_loader = dataloader['test']

    _, inputs, _ = next(test_data)

    model_outputs, encoder_cont, encoder_q = model(inputs.to(device))

    tokens = model.codebook(encoder_cont)
    tokens.reshape(tokens.shape[0], -1)
    import pdb; pdb.set_trace()
