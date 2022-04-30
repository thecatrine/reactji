from loaders import twitch
from models import diffuser

import random
import torch
import torchvision

import matplotlib.pyplot as plt


from models import diffuser
import argparse

random.seed("fooo")

p = argparse.ArgumentParser()
p.add_argument('--resume', default="")

args = p.parse_args()


def batch_generator(batch_size, max_timesteps=150):
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

                num_steps = random.randint(0, max_timesteps)
                noised_image_output = twitch.noise_img(current_image, num_steps)
                noised_image_input = twitch.noise_img(noised_image_output, 1)

                batch_steps.append(num_steps)
                batch_inputs.append(noised_image_input.unsqueeze(0))
                batch_outputs.append(noised_image_output.unsqueeze(0))

            yield (torch.tensor(batch_steps), torch.cat(batch_inputs), torch.cat(batch_outputs))
    except StopIteration as e:
        pass


gen = batch_generator(batch_size=10)

steps, inputs, outputs = next(gen)

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

model = diffuser.Diffuser()
if args.resume:
    print("Resuming from {}".format(args.resume))
    model.load_state_dict(torch.load(args.resume))
model.to(device)
#model.to(device)

loss_fn = torch.nn.MSELoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

BATCH_SIZE = 128

def train_one_epoch(epoch_index):
    running_loss = 0.
    last_loss = 0.

    i = 0
    for data in batch_generator(BATCH_SIZE):
        i += 1
        s, inputs, expected_outputs = data
        s, inputs, expected_outputs = s.to(device), inputs.to(device), expected_outputs.to(device)
        
        optimizer.zero_grad()
        outputs = model(s, inputs)
        loss = loss_fn(outputs, expected_outputs)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 1000
            print("  batch {} loss: {}".format(i + 1, last_loss))
            tb_x = epoch_index * 1000000 + i + 1
            print('Loss/train', last_loss, tb_x)
            print('Loss of identity:', loss_fn(inputs, expected_outputs).item())
            running_loss = 0.
    
    return last_loss

# Train it

EPOCHS = 10
best_vloss = 10e10

for epoch in range(EPOCHS):
    print("Epoch {}".format(epoch + 1))
    model.train(True)
    train_loss = train_one_epoch(epoch)

    model.train(False)

    running_vloss = 0.
    validation = batch_generator(BATCH_SIZE)
    for j in range(20): # I dunno do some batches
        vdata = next(validation)
        s, vinputs, vlabels = vdata
        s, vinputs, vlabels = s.to(device), vinputs.to(device), vlabels.to(device)
        voutputs = model(s, inputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss.item()
    
    avg_vloss = running_vloss / 1000000
    print('LOSS train {} valid {}'.format(train_loss, avg_vloss))

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        torch.save(model.state_dict(), 'best_model.pth')