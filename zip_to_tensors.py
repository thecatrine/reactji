from multiprocessing import dummy

import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from loaders import datasets
from models import diffuser
from loaders import loader_utils
import random

import tarfile
import zipfile
import os
import sys
import logging
import argparse
import io
import PIL.Image as Image

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)
log = logging.getLogger(__name__)

def render_batch(*args):
    foos = []
    for arg in args:
        foos.append(torchvision.utils.make_grid(arg))

    image_foos = []
    res = torch.cat(foos, dim=1)
    res = loader_utils.tensor_to_image(res)

    plt.imshow(res)
    plt.savefig("batch.png", dpi=1000)


file = tarfile.open("loaders/data/small/valid_32x32.tar", "r")

import pdb; pdb.set_trace()

namelist = z_file.namelist()
tensors = []
wrong_shape = 0
errors = 0

for filename in namelist:
        try:
            image_data = z_file.read(filename)
        except Exception as e:
            print(f'Error loading image {filename}: {e}')
            sys.exit(1)

        try:
            image = Image.open(io.BytesIO(image_data))
            image = image.convert('RGBA')
            tensor = loader_utils.image_to_tensor(image)
            if tensor.shape != (3, 28, 28):
                print(f'Image {filename} has dimensions {tensor.shape}')
                wrong_shape += 1
            else:
                tensors.append(tensor.unsqueeze(0))
        except Exception as e:
            print(f'Error parsing image {filename}: {e}')
            errors += 1


all_tensors = torch.cat(tensors, dim=0)
print("Final tensor dimensions:", all_tensors.shape)
print("Wrong shape:", wrong_shape, "out of", len(namelist))
print("Errors:", errors, "out of", len(namelist))
all_tensors.save('all_tensors.pt')
