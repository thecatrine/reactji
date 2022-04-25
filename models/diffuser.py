import torch
import torch.nn as nn
import torch.nn.functional as F

OUT_CHANNELS = 16

class Diffuser(torch.nn.Module):
    def __init__(self, channels=32, num_attentions=3, num_heads=1):
        super(Model, self).__init__()
        # input is b x 3 x 28 x 28

        # work out math on this later
        self.conv1 = nn.Conv1d(3, channels, 1)
        
        self.attention_blocks = []
        self.residual_blocks = []

        for i in range(num_attentions):
            self.attention_blocks.append(nn.MultiheadAttention(channels, num_heads))
            self.residual_blocks.append(nn.Conv1d(channels, channels, 1))

        self.final_conv = nn.Conv1d(channels, 3, 1)

    def forward(self, x):
        batch_len, init_channels, height, width = x.shape

        y = x.reshape(batch_len, init_channels, -1)
        y2 = F.SiLU(self.conv1(y))
        
        intermediate = y2
        for i in range(len(self.attention_blocks)):
            attention_block = self.attention_blocks[i]
            residual_block = self.residual_blocks[i]

            attn = F.SiLU(attention_block(intermediate))
            # why not attn? 
            # do we SiLU both things?
            resid = F.SiLU(residual_block(attn) + intermediate)

            intermediate = resid

        final = self.final_conv(intermediate)

        return final.reshape(batch_len, 3, height, width)