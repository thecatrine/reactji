import torchvision
import numpy as np
import torch


def noise_img(img, n=1, alpha=0.95):
    return torch.normal(np.sqrt(alpha**n)*img, (1-alpha**n))

SCALE = 10
whitener = whiten.Whitener(28, 28)
def image_to_tensor(im):
    tensor = torchvision.transforms.ToTensor()(im)

    scaled_tensor = (tensor - 0.5) * SCALE
    return whitener.transform(scaled_tensor)

def tensor_to_image(tensor):
    tensor = whitener.untransform(tensor)
    tensor = (tensor / SCALE) + 0.5
    tensor = torch.clip(input=tensor, min=0, max=1)

    image = torchvision.transforms.ToPILImage()(tensor)
    return image
