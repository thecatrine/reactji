from loaders import twitch
from models import diffuser

import random
import torch
import torchvision

import matplotlib.pyplot as plt

random.seed("foo")


def batch_generator(batch_size, max_timesteps=2):
    image_generator = twitch.get_from_zip()

    # batch of inputs, batch of outputs
    batch_steps = []
    batch_inputs = []
    batch_outputs = []

    for i in range(batch_size):
        current_image = next(image_generator)

        num_steps = random.randint(0, max_timesteps)
        noised_image_output = twitch.noise_img(current_image, num_steps)
        noised_image_input = twitch.noise_img(noised_image_output, 1)

        batch_steps.append(num_steps)
        batch_inputs.append(noised_image_input.unsqueeze(0))
        batch_outputs.append(noised_image_output.unsqueeze(0))

    yield (torch.tensor(batch_steps), torch.cat(batch_inputs), torch.cat(batch_outputs))


gen = batch_generator(batch_size=5)

steps, inputs, outputs = next(gen)

def render_batch(batch_in, batch_out):
    grid_in = torchvision.utils.make_grid(batch_in)
    grid_out = torchvision.utils.make_grid(batch_out)

    res = torch.cat((grid_in, grid_out), dim=1)
    res = twitch.tensor_to_image(res)

    plt.imshow(res)
    plt.show()

render_batch(inputs, outputs)