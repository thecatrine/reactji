import torchvision
import numpy as np
import torch
import math
from . import whiten

def noise_img(img, n=1, alpha=0.9999):
    return torch.normal(np.sqrt(alpha**n)*img, np.sqrt(1-alpha**n))

def weighted_timestep(max_ts=1000):
    return math.floor((np.random.random() * max_ts**0.5)**2)

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
