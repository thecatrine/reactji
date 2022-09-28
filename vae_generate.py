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

def save_images(images, emoji_prefix=None):
    os.makedirs(emoji_prefix, exist_ok=False)
    if emoji_prefix is not None:
        for i in range(len(images)):
            res = loader_utils.tensor_to_image(images[i].cpu())
            res.save(f"{emoji_prefix}/emoji-{i}.png")

device = torch.device('cuda')

hidden_size = 256
k = 512
batch_size = 128
num_epochs = 10000
lr = 2e-4
beta = 1.0

data = datasets.TwitchData(batch_size=256, max_ts=1)
dataloaders = data.dataloaders()
test_data = iter(dataloaders['test'])

model = VectorQuantizedVAE(3, hidden_size, k).to(device)

model.load_state_dict(torch.load('vqvae/model_1159.pt'))

model.eval()

_, inputs, _ = next(test_data)

model_outputs, encoder_cont, encoder_q = model(inputs.to(device))

save_images(inputs, 'vqvae/original')
save_images(model_outputs.detach(), 'vqvae/reconstructed')

test_vectors = torch.randint(0, 512, (18, 7, 7)).to(device)
test_quant = model.decode(test_vectors)

save_images(test_quant.detach(), 'vqvae/test')

import pdb; pdb.set_trace()

