from PIL import Image
import logging
from torch.utils.data import Dataset, DataLoader
import hashlib
import io
import numpy as np
import random
import torch
import torchvision
import zipfile
from .loader_utils import image_to_tensor, noise_img, max_ts

logger = logging.getLogger(__name__)

class TwitchDataset(Dataset):
    def __init__(self, z_file, namelist, max_ts=1000, alpha=0.99):
        super().__init__()
        self.max_ts = max_ts
        self.alpha = 0.99
        self.z_file = z_file
        self.namelist = namelist

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, idx):
        assert idx < len(self.namelist)
        filename = self.namelist[idx]
        try:
            image_data = self.z_file.read(filename)
        except Exception as e:
            logger.debug(f'Error loading image {filename}: {e}')
            return None

        try:
            image = Image.open(io.BytesIO(image_data))
            image = image.convert('RGBA')
            tensor = image_to_tensor(image)
            if tensor.shape != (3, 28, 28):
                logger.debug(f'Image {filename} has dimensions {tensor.shape}')
                return None
            steps = weighted_timestep(max_ts)
            img_out = noise_img(tensor, steps, self.alpha)
            img_in = noise_img(img_out, 1, self.alpha)
            return (torch.tensor(steps), img_in, img_out)
        except Exception as e:
            logger.debug(f'Error parsing image {filename}: {e}')
            return None

class TwitchData():
    def __init__(self, path='loaders/data/twitch_archive.zip',
                 batch_size=128, shuffle=True, num_workers=8):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.z_file = zipfile.ZipFile(path, 'r')
        self.full_namelist = self.z_file.namelist()

    def dataloaders(self):
        namelists = {}
        dataloaders = {}
        splits = [('val', b'\x02'), ('test', b'\x04'), ('train', None)]

        for name in self.full_namelist:
            name_hash = hashlib.sha256(name.encode('utf-8')).digest()
            for split, breakpoint in splits:
                if breakpoint is None or name_hash[:1] < breakpoint:
                    namelists.setdefault(split, []).append(name)

        default_collate = torch.utils.data.dataloader.default_collate
        def collate_fn(batch):
            batch = [x for x in batch if x is not None]
            return default_collate(batch)

        for split, namelist in namelists.items():
            dataset = TwitchDataset(self.z_file, namelist)
            dataloaders[split] = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                num_workers=self.num_workers, collate_fn=collate_fn,
            )

        return dataloaders

    def __enter__(self):
        pass

    def __exit__(self):
        self.z_file.close()
