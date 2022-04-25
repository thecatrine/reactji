import zipfile
import matplotlib
import matplotlib.pyplot as plt
import io

from PIL import Image
import numpy as np
import torch
import torchvision

import random

SEED = 12345
# Seed random number generator
random.seed(SEED)

def get_from_zip():
    with zipfile.ZipFile('loaders/data/twitch_archive.zip', 'r') as zip:
        print("Archive loaded")
        namelist = zip.namelist()
        random.shuffle(namelist)
        for filename in namelist:
            print("opening file: " + filename)
            image_data = zip.read(filename)

            image = Image.open(io.BytesIO(image_data))
            yield image_to_tensor(image.convert("RGB"))

def noise_img(img, n=1, alpha=0.9985):                    
    return torch.normal(np.sqrt(alpha**n)*img, (1-alpha**n))

SCALE = 10

def image_to_tensor(im):
    tensor = torchvision.transforms.ToTensor()(im)

    scaled_tensor = ((tensor - 128) / 256) * SCALE
    return scaled_tensor

def tensor_to_image(tensor):
    tensor = (tensor * 256 / SCALE) + 128

    image = torchvision.transforms.ToPILImage()(tensor)
    return image

if __name__=="__main__":
    image_generator = get_from_zip()

    for i in range(10):
        scaled_tensor = next(image_generator)

        std = 0.001

        images = []

        for i in range(0, 10):
            images.append(noise_img(scaled_tensor, i))


        to_render = [tensor_to_image(foo) for foo in images]

        plt.figure(figsize=(10, 10))
        for i in range(len(to_render)):
            plt.subplot(1, len(to_render), i + 1)
            plt.imshow(to_render[i])
        plt.show()