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
from transformers import CLIPProcessor, CLIPModel

device = torch.device('cuda')

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
for param in clip_model.parameters():
    param.requires_grad = True
clip_model.train().to(device)

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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
p.add_argument('--clip-guidance', default='', type=str)

args = p.parse_args()

print(args.clip_guidance)

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


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

class MakeCutouts(torch.nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)

if True:
    # Generate diffusion
    for i in range(STAPS, 0, -STAP_SIZE):

        s = torch.Tensor([i])
        with torch.no_grad():
            outputs = generate_model.forward(temp.to(device), s.to(device))
        orig = temp - outputs

        if args.clip_guidance:
            foo = orig.clone().detach().cpu().requires_grad_(True)

            foo_image = torch.stack([loader_utils.unwhiten_differentiable(f) for f in foo])
            #import pdb; pdb.set_trace()
            foo2 = torch.nn.functional.interpolate(foo_image, size=(224, 224), mode='nearest')

            clip_processor.feature_extractor.do_resize = False
            clip_processor.feature_extractor.do_center_crop = False
            clip_processor.feature_extractor.do_convert_rgb = False

            #make_cutouts = MakeCutouts(224, 16)

            #foo_batch = make_cutouts(foo2)

            foo_list=[f for f in foo2]
            
            #import pdb; pdb.set_trace()

            clip_inputs = clip_processor(text=[args.clip_guidance], images=foo_list, return_tensors=None)
            clip_inputs['pixel_values'] = torch.stack(clip_inputs['pixel_values']).to(device)
            clip_inputs['input_ids'] = torch.tensor(clip_inputs['input_ids']).to(device)
            clip_inputs['attention_mask'] = torch.tensor(clip_inputs['attention_mask']).to(device)

            clip_outputs = clip_model(**clip_inputs)

            # Why is this the thing to do?
            # Spherical distance loss?
            text_embeds = clip_outputs.text_embeds.repeat(clip_outputs.image_embeds.shape[0], 1) 
            image_embeds = clip_outputs.image_embeds

            #import pdb; pdb.set_trace()

            loss = spherical_dist_loss(text_embeds, image_embeds).sum()
            guidance = -torch.autograd.grad(loss, foo)[0].to(device).detach()
            print("guidance", loss.item())
            # add gradient to temp
            orig = orig

        for j in range(i, i-STAP_SIZE, -1):
            temp = loader_utils.take_step(temp, orig, j, condition=guidance)

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
        with torch.no_grad():
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
