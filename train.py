from loaders import twitch
from models import diffuser

import random
import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

import zipfile
from models import diffuser
import argparse

random.seed("fooo")

p = argparse.ArgumentParser()
p.add_argument('--resume', default="")

args = p.parse_args()


def batch_generator(image_generator, batch_size, max_timesteps=150):
    # batch of inputs, batch of outputs

    go = True

    while go:
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
                return
            except Exception as e:
                print("Skipping because of exception")
                print(e)
                print("---")
                continue
            i += 1

            num_steps = random.randint(0, max_timesteps)
            noised_image_output = twitch.noise_img(current_image, num_steps)
            noised_image_input = twitch.noise_img(noised_image_output, 1)

            batch_steps.append(num_steps)
            batch_inputs.append(noised_image_input.unsqueeze(0))
            batch_outputs.append(noised_image_output.unsqueeze(0))

        yield (torch.tensor(batch_steps), torch.cat(batch_inputs), torch.cat(batch_outputs))


def render_batch(batch_in, batch_out):
    grid_in = torchvision.utils.make_grid(batch_in)
    grid_out = torchvision.utils.make_grid(batch_out)

    res = torch.cat((grid_in, grid_out), dim=1)
    res = twitch.tensor_to_image(res)

    plt.imshow(res)
    plt.show()

#render_batch(inputs, outputs)

# GPU device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

## Attempt training

model = diffuser.Diffuser(dropout_rate=0.1, normalization_groups=32)
if args.resume:
    print("Resuming from {}".format(args.resume))
    model.load_state_dict(torch.load(args.resume))
model.to(device)

loss_fn = torch.nn.MSELoss()

# optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

BATCH_SIZE = 100

writer = SummaryWriter()

def train_one_epoch(training_batch_generator, total_len, epoch_index):
    running_loss = 0.
    last_loss = 0.

    i = 0

    for data in training_batch_generator:
        i += 1
        s, inputs, expected_outputs = data
        s, inputs, expected_outputs = s.to(device), inputs.to(device), expected_outputs.to(device)
        
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
            writer.add_scalar('Loss/identity', loss_fn(inputs, expected_outputs).item(), tb_x)
            running_loss = 0.

            #return last_loss
    
    return last_loss

# Train it

EPOCHS = 100
best_vloss = 10e10

z_file = zipfile.ZipFile('loaders/data/twitch_archive.zip', 'r')

for epoch in range(EPOCHS):
    print("Epoch {}".format(epoch + 1))

    N_train, training_image_generator, N_test, test_image_generator, N_validate, validation_image_generator = twitch.get_from_zip(z_file)

    # Train an epoch
    training_batch_generator = batch_generator(training_image_generator, BATCH_SIZE)
    model.train(True)
    train_loss = train_one_epoch(training_batch_generator, N_train, epoch)
    model.train(False)

    # Test with some batches
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
        torch.save(model.state_dict(), f"best_model_{epoch}.pth")
    torch.save(model.state_dict(), f"cur_model.pth")