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

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)
log = logging.getLogger(__name__)



p = argparse.ArgumentParser()
p.add_argument('--generate-model', default="")
p.add_argument('--upsample-model', default="")
p.add_argument('--noise', action='store_true')
p.add_argument('--emoji', default="")
p.add_argument('--refine', default="")
p.add_argument('--num', default=8, type=int)
p.add_argument('--staps', default=50, type=int)
p.add_argument('--stap-size', default=1, type=int)

args = p.parse_args()

# RSI defaults file
generate_opts = {
    'normalization_groups': 32,
    'channels': 256,
    'num_head_channels': 64,
    'num_residuals': 6,
    'channel_multiple_schedule': [1, 2, 3],
    'interior_attention': 1,
    'in_channels': 3,
}
upsample_opts = {
    'normalization_groups': 4,
    'channels': 32,
    'num_head_channels': 8,
    'num_residuals': 6,
    'channel_multiple_schedule': [1, 2, 3, 6, 12],
    'interior_attention': 3,
    'in_channels': 6,
}

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

generate_model = diffuser.Diffuser(**generate_opts)
upsample_model = diffuser.Diffuser(**upsample_opts)
device = torch.device('cuda')

generate_model.load_state_dict(torch.load(args.generate_model, map_location=device)['model'])
generate_model = generate_model.to(device)
generate_model.eval()

upsample_model.load_state_dict(torch.load(args.upsample_model, map_location=device)['model'])
upsample_model = upsample_model.to(device)
upsample_model.eval()

#
STAPS=args.staps
STAP_SIZE = args.stap_size


data = datasets.TwitchData(batch_size=256, max_ts=1)
dataloaders = data.dataloaders()
test_data = iter(dataloaders['test'])

images = []
large_images = []
s, inputs, outputs = next(test_data)

all_images = []
for i in range(args.num):
    if args.noise:
        images.append(torch.normal(torch.zeros(3, 28, 28), 1).unsqueeze(0))
        large_images.append(torch.normal(torch.zeros(3, 112, 112), 1).unsqueeze(0))
    elif args.refine:
        img = Image.open(args.refine)
        tensor = loader_utils.image_to_tensor(img)
        images.append(loader_utils.noise_img(tensor, STAPS).unsqueeze(0))
    else:
        images.append(loader_utils.noise_img(outputs[i], STAPS).unsqueeze(0))

# Try to generate something


def save_images(images, emoji_prefix=None):
    os.makedirs(emoji_prefix, exist_ok=False)
    if emoji_prefix is not None:
        for i in range(len(images)):
            res = loader_utils.tensor_to_image(images[i].cpu())
            res.save(f"{emoji_prefix}/emoji-{i}.png")


temp = torch.cat(images, dim=0).to(device)
begin = outputs[:args.num]

with torch.no_grad():
    # Generate diffusion
    for i in range(STAPS, 0, -STAP_SIZE):
        s = torch.Tensor([i])
        outputs = generate_model.forward(temp.to(device), s.to(device))
        orig = temp - outputs
        for j in range(i, i-STAP_SIZE, -1):
            temp = loader_utils.take_step(temp, orig, j)

        print(f"Generate {i}")
        if not args.emoji and (i % 50 == STAP_SIZE):
            all_images.append(temp.cpu())

    save_images(temp.cpu(), emoji_prefix=args.emoji + "-small")
    blurred = temp#torchvision.transforms.functional.gaussian_blur(temp, kernel_size=3)
    #save_images(blurred.cpu(), emoji_prefix=args.emoji + "-small-blurred")

    large_blurred = F.interpolate(temp, scale_factor=4, mode='nearest')
    #save_images(large_blurred.cpu(), emoji_prefix=args.emoji + "-large-blurred")

    conditioning = F.interpolate(blurred, scale_factor=4, mode='nearest')

    temp = torch.cat(large_images, dim=0).to(device)
    # Upsample diffusion
    for i in range(STAPS, 0, -STAP_SIZE):
        s = torch.Tensor([i])
        outputs = upsample_model.forward(torch.cat([temp, conditioning], dim=1), s.to(device))
        
        orig = temp - outputs
        for j in range(i, i-STAP_SIZE, -1):
            temp = loader_utils.take_step(temp, orig, j)

        print(f"Upsample {i}")
        if not args.emoji and (i % 50 == STAP_SIZE):
            all_images.append(temp.cpu())
    
    save_images(temp.cpu(), emoji_prefix=args.emoji)

#import pdb; pdb.set_trace()
if not args.noise and not args.refine:
    all_images.append(begin)

if args.emoji:
    all_images.append(temp.cpu())

print("all_images:", len(all_images), all_images[0].shape)
render_batch(*all_images)


#vals = model.forward(images)
#guesses = torch.argmax(vals, dim=1)

#print(vals)
#print(guesses)
#print("Guesses: ", "    ".join([fashion.classes[j] for j in guesses]))
#print("Real   : ", "    ".join(real_labels))
#fashion.matplotlib_imshow(img_gri, one_channel=True)
