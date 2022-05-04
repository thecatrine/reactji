from multiprocessing import dummy

from black import validate_metadata
import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from models import diffuser
from loaders import twitch
import random
import zipfile
import argparse

p = argparse.ArgumentParser()
p.add_argument('--model', default="")

args = p.parse_args()

def render_batch(*args):
    foos = []
    for arg in args:
        foos.append(torchvision.utils.make_grid(arg))

    image_foos = []
    res = torch.cat(foos, dim=1)
    res = twitch.tensor_to_image(res)

    plt.imshow(res)
    plt.savefig("batch.png", dpi=1000)

#render_batch(inputs, outputs)

print("Loading models")
model = diffuser.Diffuser(dropout_rate=0.1, normalization_groups=32)
model.load_state_dict(torch.load(args.model))
model.eval()

#
STAPS=150

#z_file = zipfile.ZipFile('loaders/data/twitch_archive.zip', 'r')
#n_train, training_image_generator, n_test, test_image_generator, n_valid, validation_image_generator = twitch.get_from_zip(z_file)
images = []
for i in range(8):
    #image = next(test_image_generator)
    #images.append(twitch.noise_img(image, STAPS).unsqueeze(0))
    images.append(torch.normal(torch.zeros(3, 28, 28), 1).unsqueeze(0))
# Try to generate something

all_images = []

temp = torch.cat(images, dim=0)
with torch.no_grad():
    for i in range(STAPS, 0, -1):
        s = torch.Tensor([i-1])
        outputs = model.forward(temp, s)
        print(i, "shape: ", outputs.shape)
        if i % 5 == 1:
            all_images.append(outputs)
        temp = outputs
    
    twitch.tensor_to_image(temp[0]).save('emoji-test.png')

render_batch(*all_images)

#vals = model.forward(images)
#guesses = torch.argmax(vals, dim=1)

#print(vals)
#print(guesses)
#print("Guesses: ", "    ".join([fashion.classes[j] for j in guesses]))
#print("Real   : ", "    ".join(real_labels))
#fashion.matplotlib_imshow(img_gri, one_channel=True)