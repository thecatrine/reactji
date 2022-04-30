import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils

OUT_CHANNELS = 16
class Diffuser(torch.nn.Module):
    def __init__(self, device=None, channels=32, timestamp_channels=10, num_attentions=8, num_heads=1):
        super(Diffuser, self).__init__()
        # input is b x 3 x 28 x 28
        self.channels=channels
        self.timestamp_channels=timestamp_channels

        # work out math on this later
        self.conv1 = nn.Conv1d(3+self.timestamp_channels, self.channels, 1) # why do we want more channels here?
        self.queryList = nn.ModuleList([])
        self.keyList = nn.ModuleList([])
        self.valueList = nn.ModuleList([])

        self.attention_blocks = nn.ModuleList([])
        self.residual_blocks = nn.ModuleList([])

        print("Embedding dimensions:", channels)
        # With this list of items how does it magic the backwards pass?

        for i in range(num_attentions):
            self.queryList.append(nn.Conv1d(channels, channels, 1))
            self.keyList.append(nn.Conv1d(channels, channels, 1))
            self.valueList.append(nn.Conv1d(channels, channels, 1))
            # Batch first? Who tf knows
            self.attention_blocks.append(nn.MultiheadAttention(28*28, num_heads, batch_first=True))
            # What is the embedding dimension here?
            self.residual_blocks.append(nn.Conv1d(channels, channels, 1))

        self.final_conv = nn.Conv1d(channels, 3, 1).to(device)


    def forward(self, steps, x):
        batch_len, init_channels, height, width = x.shape

        # sinusoidal timestep embedding
        timestamp_embedding = utils.timestep_embedding(steps, self.timestamp_channels)

        timestamp_embedding = torch.broadcast_to(timestamp_embedding.unsqueeze(2), (batch_len, self.timestamp_channels, height*width))

        data = x.reshape(batch_len, init_channels, -1)

        data_and_timestamps = torch.cat((data, timestamp_embedding), dim=1)

        y2 = F.silu(self.conv1(data_and_timestamps))
        
        # TODO: Embed the step somewhow
        intermediate = y2
        for i in range(len(self.attention_blocks)):
            attention_block = self.attention_blocks[i]
            residual_block = self.residual_blocks[i]

            query = self.queryList[i]
            key = self.keyList[i]
            value = self.valueList[i]

            attn_output, attn_output_weights = attention_block(query(intermediate), key(intermediate), value(intermediate))

            attn = F.silu(attn_output)
            # why not attn? 
            # do we SiLU both things?
            resid = F.silu(residual_block(attn) + intermediate)

            intermediate = resid

        final = self.final_conv(intermediate)

        return final.reshape(batch_len, 3, height, width)