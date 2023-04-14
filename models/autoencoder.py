import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from . import utils
from . import vqvae

log = logging.getLogger(__name__)

OUT_CHANNELS = 16


class Autoencoder(torch.nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # input is b x in_channels x 28 x 28

        self.down_layers = nn.ModuleList([])
        self.middle_layer = None
        self.up_layers = nn.ModuleList([])

        self.middle_layer = 
 