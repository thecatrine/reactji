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

#LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
#logging.basicConfig(level=LOGLEVEL)
#log = logging.getLogger(__name__)

a = list(range(1_400_000))
random.shuffle(a)
a = torch.tensor(a)
a = a.reshape(140, -1)

BATCH_SIZE = 10_000

for batch in range(0, 140):
    numbers = a[batch]
    batch_tensors = []
    batch_errors = 0
    batch_size_errors = 0
    for i in range(numbers.shape[0]):
        number = numbers[i]
        f = f"download/batch_{torch.div(number, BATCH_SIZE, rounding_mode='floor')}/{number}.png"
        #print(f)
        #import pdb; pdb.set_trace()
        if i % 1000 == 999:
            print('.', end='')
            sys.stdout.flush()

        try:
            im = Image.open(f)
            im = im.convert('RGBA')

            if im.size != (112, 112):
                batch_size_errors += 1
                continue

            tensor = loader_utils.image_to_tensor(im)

            #import pdb; pdb.set_trace()

            batch_tensors.append(tensor)

        except Exception as e:
            if random.random() < 0.01:
                print(e)
            batch_errors += 1


    all_batch_tensors = torch.cat(batch_tensors, dim=0)

    print("")
    print(f"Batch {batch} errors: {batch_errors}")
    print(f"Batch {batch} sizing issues: {batch_size_errors}")
    torch.save(all_batch_tensors, f'download/batch_tensors_{batch}.pt')
