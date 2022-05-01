import zipfile
import matplotlib
import matplotlib.pyplot as plt
import io

from PIL import Image
import numpy as np
import torch
import torchvision

import random
import hashlib

SEED = 12345
# Seed random number generator
random.seed(SEED)

def namelist_to_generator(zipfile, namelist):
    for filename in namelist:
        try:
            image_data = zipfile.read(filename)
        except:
            print("Zipfile read error")
            break

        try:
            image = Image.open(io.BytesIO(image_data))
            yield image_to_tensor(image.convert("RGB"))
        except:
            print("Error loading image: " + filename)
            continue

def get_from_zip(z_file):
    print("Archive loaded")
    namelist = z_file.namelist()
    random.shuffle(namelist)

    train_names = []
    test_names = []
    validation_names = []

    for name in namelist:
        name_hash = hashlib.sha256(name.encode('utf-8')).digest()
        if name_hash[:1] < b"\x02":
            validation_names.append(name)
        elif name_hash[:1] < b"\x04":
            test_names.append(name)
        else:
            train_names.append(name)

    print("Training:", len(train_names))
    print("Testing:", len(test_names))
    print("Validation:", len(validation_names))

    # return three generators
    return len(train_names), namelist_to_generator(z_file, train_names), len(test_names), namelist_to_generator(z_file, test_names), len(validation_names), namelist_to_generator(z_file, validation_names)

def noise_img(img, n=1, alpha=0.95):                    
    return torch.normal(np.sqrt(alpha**n)*img, (1-alpha**n))

SCALE = 10

def image_to_tensor(im):
    tensor = torchvision.transforms.ToTensor()(im)

    scaled_tensor = (tensor - 0.5) * SCALE
    return scaled_tensor

def tensor_to_image(tensor):
    tensor = (tensor / SCALE) + 0.5
    tensor = torch.clip(input=tensor, min=0, max=1)

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