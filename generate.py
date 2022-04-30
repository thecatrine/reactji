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

def batch_generator(batch_size, max_timesteps=1):
    image_generator = twitch.get_from_zip()

    # batch of inputs, batch of outputs

    try:
        while True:
            batch_steps = []
            batch_inputs = []
            batch_outputs = []

            i = 0
            while i < batch_size:
                try:
                    current_image = next(image_generator)
                    if current_image.shape != (3, 28 , 28):
                        print("Skipping because of shape: ", current_image.shape)
                        continue
                except StopIteration:
                    print("Reached end of archive")
                    raise
                except Exception as e:
                    print("Skipping because of exception")
                    print(e)
                    print("---")
                    continue
                i += 1

                # TODO: num steps that isn't just 1
                num_steps = 0 # random.randint(0, max_timesteps)
                noised_image_output = twitch.noise_img(current_image, num_steps)
                noised_image_input = twitch.noise_img(noised_image_output, num_steps+1)

                batch_steps.append(num_steps)
                batch_outputs.append(noised_image_output.unsqueeze(0))
                batch_inputs.append(noised_image_input.unsqueeze(0))

            yield (torch.tensor(batch_steps), torch.cat(batch_inputs), torch.cat(batch_outputs))
    except StopIteration as e:
        pass

def render_batch(*args):
    foos = []
    for arg in args:
        foos.append(torchvision.utils.make_grid(arg))

    res = torch.cat(foos, dim=1)
    res = twitch.tensor_to_image(res)

    plt.imshow(res)
    plt.savefig("batch.png", dpi=1000)

#render_batch(inputs, outputs)

print("Loading models")
model = diffuser.Diffuser()
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

#
gen = twitch.get_from_zip()
images = []
for i in range(8):
    image = next(gen)
    images.append(twitch.noise_img(image, 75).unsqueeze(0))
# Try to generate something

all_images = []

temp = torch.cat(images, dim=0)
for i in range(75, 0, -1):
    s = torch.Tensor([i])
    outputs = model.forward(s, temp)
    print("shape: ", outputs.shape)
    all_images.append(outputs)
    temp = outputs

render_batch(*all_images)

#vals = model.forward(images)
#guesses = torch.argmax(vals, dim=1)

#print(vals)
#print(guesses)
#print("Guesses: ", "    ".join([fashion.classes[j] for j in guesses]))
#print("Real   : ", "    ".join(real_labels))
#fashion.matplotlib_imshow(img_gri, one_channel=True)