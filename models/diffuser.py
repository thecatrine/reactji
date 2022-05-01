import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils

OUT_CHANNELS = 16

class AttentionResid(nn.Module):
    def __init__(self, dims, channels, num_heads=1):
        super(AttentionResid, self).__init__()

        self.channels = channels
        self.dims = dims
        total_dim = dims[0]*dims[1]

        self.query_conv = nn.Conv1d(self.channels, self.channels, 1)
        self.key_conv = nn.Conv1d(self.channels, self.channels, 1)
        self.value_conv = nn.Conv1d(self.channels, self.channels, 1)

        self.attention_block = nn.MultiheadAttention(total_dim, num_heads, batch_first=True)
        self.residual_block = nn.Conv2d(self.channels, self.channels, 3, padding=1)
        pass

    def forward(self, x):
        batch_len, channels, dims = x.shape
        ydim, xdim = self.dims
        
        y, weights = self.attention_block(self.query_conv(x), self.key_conv(x), self.value_conv(x))

        y = F.silu(y)
        y = y.reshape(batch_len, self.channels, ydim, xdim)
        
        # TODO: This feels like a weird place to put the conv2d, and a weird name for it
        y = self.residual_block(y)
        y = y.reshape(batch_len, self.channels, -1)

        # Add back in the original channels
        return F.silu(y + x)
class Diffuser(torch.nn.Module):
    def __init__(self, device=None, channels=32, timestamp_channels=10, num_attentions=8, num_heads=1):
        super(Diffuser, self).__init__()
        # input is b x 3 x 28 x 28
        self.channels=channels
        self.timestamp_channels=timestamp_channels

        # work out math on this later
        self.conv1 = nn.Conv1d(3+self.timestamp_channels, self.channels, 1) # why do we want more channels here?

        self.attention_resid_blocks = nn.ModuleList([])

        print("Embedding dimensions:", channels)
        # With this list of items how does it magic the backwards pass?

        for i in range(num_attentions):
            self.attention_resid_blocks.append(AttentionResid((28, 28), channels))

        self.final_conv = nn.Conv1d(channels, 3, 1).to(device)

    def forward(self, steps, x):
        batch_len, init_channels, height, width = x.shape

        # sinusoidal timestep embedding
        timestamp_embedding = utils.timestep_embedding(steps, self.timestamp_channels)

        timestamp_embedding = torch.broadcast_to(timestamp_embedding.unsqueeze(2), (batch_len, self.timestamp_channels, height*width))

        data = x.reshape(batch_len, init_channels, -1)

        data_and_timestamps = torch.cat((data, timestamp_embedding), dim=1)

        y2 = F.silu(self.conv1(data_and_timestamps))
        
        intermediate = y2
        for i in range(len(self.attention_resid_blocks)):
            attention_resid = self.attention_resid_blocks[i]
            intermediate = attention_resid(intermediate)

        final = self.final_conv(intermediate)

        return final.reshape(batch_len, 3, height, width)