from loaders import datasets

import torch

old_twitch = datasets.TwitchData(max_ts=0, num_workers=0)

dataloaders = old_twitch.dataloaders()


for kind in ['test', 'val']:
    train = dataloaders[kind]

    all_tensors = []
    for item in train:
        all_tensors.append(item[2])
        print(len(all_tensors), len(train))

    all_tensors = torch.cat(all_tensors)

    torch.save(all_tensors, 'loaders/data/twitch/'+kind+'.pt')
