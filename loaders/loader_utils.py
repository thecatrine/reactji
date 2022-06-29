import torchvision
import numpy as np
import torch
import math
from . import whiten

MAX_TS = 1000
MIN_BETA=0.0001
MAX_BETA=0.02
BETAS = []
for i in range(MAX_TS):
    BETAS.append(MIN_BETA + (i/MAX_TS)*(MAX_BETA-MIN_BETA))
BETAS = torch.tensor(BETAS)
ALPHAS = 1-BETAS
acc = 1
ALPHAS_CUMPROD = [acc]
for i in range(len(ALPHAS)):
    acc *= ALPHAS[i]
    ALPHAS_CUMPROD.append(acc)
ALPHAS_CUMPROD = torch.tensor(ALPHAS_CUMPROD)
print(ALPHAS)
print(ALPHAS_CUMPROD)

def noise_img(img, n=1):
    return torch.normal(np.sqrt(ALPHAS_CUMPROD[n])*img, np.sqrt(1-ALPHAS_CUMPROD[n]))

def weighted_timestep(max_ts=1000):
    return math.floor(np.random.random() * max_ts)

SCALE = 10
whitener = whiten.Whitener()
def image_to_tensor(im):
    tensor = torchvision.transforms.ToTensor()(im)
    if tensor.shape[0] == 4:
        tensor = tensor[:3, :, :]

    scaled_tensor = (tensor - 0.5) * SCALE
    return whitener.transform(scaled_tensor)

def tensor_to_image(tensor):
    tensor = whitener.untransform(tensor)
    tensor = (tensor / SCALE) + 0.5
    tensor = torch.clip(input=tensor, min=0, max=1)

    image = torchvision.transforms.ToPILImage()(tensor)
    return image
