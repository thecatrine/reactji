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
import zipfile
import os
import logging
import argparse

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)
log = logging.getLogger(__name__)


p = argparse.ArgumentParser()
p.add_argument('--model', default="")
p.add_argument('--noise', action='store_true')
p.add_argument('--staps', default=50, type=int)
p.add_argument('--stap-size', default=1, type=int)

args = p.parse_args()

def render_batch(*args):
    foos = []
    for arg in args:
        foos.append(torchvision.utils.make_grid(arg))

    image_foos = []
    res = torch.cat(foos, dim=1)
    res = loader_utils.tensor_to_image(res)

    plt.imshow(res)
    plt.savefig("batch.png", dpi=1000)

#render_batch(inputs, outputs)

print("Loading models")

device = torch.device('cuda')
model = diffuser.Diffuser(dropout_rate=0.1)
model.load_state_dict(torch.load(args.model, map_location=device)['model'])
model.eval()

#
STAPS=args.staps
STAP_SIZE = args.stap_size


data = datasets.TwitchData(batch_size=128, max_ts=0)
dataloaders = data.dataloaders()
test_data = iter(dataloaders['test'])

images = []
s, inputs, outputs = next(test_data)
for i in range(8):
    if args.noise:
        images.append(torch.normal(torch.zeros(3, 28, 28), 1).unsqueeze(0))
    else:
        images.append(loader_utils.noise_img(outputs[i], STAPS).unsqueeze(0))
    

# Try to generate something

all_images = []



temp = torch.cat(images, dim=0)
with torch.no_grad():
    for i in range(STAPS, 0, -1):
        s = torch.Tensor([i * STAP_SIZE - 1])
        outputs = model.forward(temp, s)
        diff = outputs - temp
        #print(diff)
        outputs = temp + diff * STAP_SIZE
        print(i, "shape: ", outputs.shape)
        if i % 10 == 1:
            all_images.append(outputs)
        temp = outputs
    
    loader_utils.tensor_to_image(temp[0]).save('emoji-test.png')

render_batch(*all_images)

#vals = model.forward(images)
#guesses = torch.argmax(vals, dim=1)

#print(vals)
#print(guesses)
#print("Guesses: ", "    ".join([fashion.classes[j] for j in guesses]))
#print("Real   : ", "    ".join(real_labels))
#fashion.matplotlib_imshow(img_gri, one_channel=True)
