import torchvision
import numpy as np
import torch
import torch.functional as F
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
ALPHAS_CUMPROD = torch.cumprod(ALPHAS, dim=0)

# THis shit is just copieed from dalle2_pytorch.py:446
ALPHAS_CUMPROD_PREV = F.pad(ALPHAS_CUMPROD[:-1], (1, 0), value = 1.)
POSTERIOR_MEAN_COEF1 = BETAS * torch.sqrt(ALPHAS_CUMPROD_PREV) / (1. - ALPHAS_CUMPROD))
POSTERIOR_MEAN_COEF2 = (1. - ALPHAS_CUMPROD_PREV) * torch.sqrt(ALPHAS) / (1. - ALPHAS_CUMPROD))
POSTERIOR_VARIANCE = BETAS * (1. - ALPHAS_CUMPROD_PREV) / (1. - ALPHAS_CUMPROD)
POSTERIOR_LOG_VARIANCE = torch.log(POSTERIOR_VARIANCE.clamp(min=1e-20)))
print(ALPHAS)
print(ALPHAS_CUMPROD)

def noise_img(img, n=1):
    assert n > 0
    return torch.normal(np.sqrt(ALPHAS_CUMPROD[n-1])*img, np.sqrt(1-ALPHAS_CUMPROD[n-1]))

def weighted_timestep(max_ts=1000):
    return math.floor(np.random.random() * max_ts)

def take_step(noised_img, predicted_true_img, n):
    mean = POSTERIOR_MEAN_COEF1[n-1] * predicted_true_image + POSTERIOR_MEAN_COEF2[n-1] * noised_img
    noise = torch.randn_like(mean)
    # TODO: Figure out whether we fucked up the logic with this zero indexing thing
    # nonzero_mask = (1 - (n == 0).float()).reshape(mean.shape[0], *((1,) * (len(mean.shape) - 1)))
    return mean + (0.5 * POSTERIOR_LOG_VARIANCE[n-1]).exp() * noise

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
