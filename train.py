from loaders import datasets
from models import diffuser
from models import diffuser
from torch.utils.tensorboard import SummaryWriter
import logging
import argparse
import matplotlib.pyplot as plt
import random
import torch
import torchvision
import zipfile
import numpy as np
import os

LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)
logger = logging.getLogger(__name__)

random.seed(0)
np.random.seed(0)

p = argparse.ArgumentParser()
p.add_argument('--resume', default="")
args = p.parse_args()


def render_batch(batch_in, batch_out):
    grid_in = torchvision.utils.make_grid(batch_in)
    grid_out = torchvision.utils.make_grid(batch_out)

    res = torch.cat((grid_in, grid_out), dim=1)
    res = twitch.tensor_to_image(res)

    plt.imshow(res)
    plt.show()



# Data
data = datasets.TwitchData(batch_size=128)

# Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")
model = diffuser.Diffuser(dropout_rate=0.1, normalization_groups=32)
if args.resume:
    logger.info("Resuming from {}".format(args.resume))
    model.load_state_dict(torch.load(args.resume))
model.to(device)
loss_fn = torch.nn.MSELoss()
# optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)


# Tensorboard
writer = SummaryWriter()

def train_one_epoch(train_data, total_len, epoch_index):
    model.train(True)
    running_loss = 0.
    last_loss = 0.

    for i, batches in enumerate(train_data):
        s, inputs, expected_outputs = batches
        s, inputs, expected_outputs = (
            s.to(device), inputs.to(device), expected_outputs.to(device))
        batch_sz = s.shape[0]

        optimizer.zero_grad()
        outputs = model(inputs, s)
        loss = loss_fn(outputs, expected_outputs)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100
            print("  batch {} loss: {}".format(i + 1, last_loss))
            tb_x = epoch_index * total_len + i*inputs.shape[0] + 1

            print('Loss/train', last_loss, tb_x)
            writer.add_scalar('Loss/train', last_loss, tb_x)

            print('Loss of identity:', loss_fn(inputs, expected_outputs).item())
            writer.add_scalar(
                'Loss/identity', loss_fn(inputs, expected_outputs).item(), tb_x)
            running_loss = 0.


    return last_loss

# Train it

EPOCHS = 100
best_vloss = 10e10

z_file = zipfile.ZipFile('loaders/data/twitch_archive.zip', 'r')

for epoch in range(EPOCHS):
    print("Epoch {}".format(epoch + 1))

    dataloaders = data.dataloaders()
    train_data = dataloaders['train']
    test_data = dataloaders['test']

    train_loss = train_one_epoch(train_data, len(train_data), epoch)
    model.train(False)
    test_batch_generator = batch_generator(test_image_generator, BATCH_SIZE)
    running_vloss = 0.

    test_len = 0
    for vdata in test_batch_generator: # I dunno do some batches
        s, vinputs, vlabels = vdata
        s, vinputs, vlabels = s.to(device), vinputs.to(device), vlabels.to(device)
        voutputs = model(vinputs, s)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss.item()
        test_len += 1

    # Try to clean up variables
    s, vinputs, vlabels = None, None, None
    voutputs = None
    vloss = None

    avg_vloss = running_vloss / test_len
    print('LOSS train {} valid {}'.format(train_loss, avg_vloss))
    writer.add_scalar('Loss/valid', avg_vloss, epoch)

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
<<<<<<< HEAD
        torch.save(model.state_dict(), f"best_model_{epoch}.pth")
||||||| d183d16
        torch.save(model.state_dict(), f"best_model_{epoch}.pth")
=======
        torch.save(model.state_dict(), f"best_model_{epoch}.pth")
    torch.save(model.state_dict(), f"cur_model.pth")
>>>>>>> 2408394cf558d9557dcef17f26677ebb02e84563
