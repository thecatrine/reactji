import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils

OUT_CHANNELS = 16

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_channels, dropout_rate, normalization_groups):
        super(Residual, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_channels = embedding_channels
        self.dropout_rate = dropout_rate
        self.normalization_groups = normalization_groups

        self.norm1 = nn.GroupNorm(normalization_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # Map embeddings into correct dimension
        # Why do we add rather than concatenate? something something information something?
        self.emb_mapping = nn.Linear(embedding_channels, out_channels)
        self.norm2 = nn.GroupNorm(normalization_groups, out_channels)

        self.droupout = nn.Dropout(p=dropout_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels == out_channels:
            self.residual_connection = nn.Identity()
        else:
            self.residual_connection = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, orig_batch, timesteps):
        # batch is [N x C x H x W]
        batch = self.norm1(orig_batch)
        batch = F.silu(batch)
        batch = self.conv1(batch)

        # Map timestamps N x C
        timesteps = self.emb_mapping(F.silu(timesteps))

        # Add timestamps into batch
        batch = batch + timestamps

        batch = F.silu(self.norm2(batch))

        batch = self.droupout(batch)
        batch = self.conv2(batch)

        resid = self.residual_connection(orig_batch)

        return batch + resid

class Attention(nn.Module):
    def __init__(self, dims, channels, normalization_groups, num_heads):
        super(AttentionResid, self).__init__()
        assert(channels % num_heads == 0)
        
        self.dims = dims
        self.channels = channels
        self.num_heads = num_heads

        self.num_head_channels = channels // num_heads

        self.norm1 = nn.GroupNorm(normalization_groups, channels)

        # Triple number of channels for q,k,v
        self.conv1 = nn.Conv1d(channels, channels*3, 1)
        
        # Do we want dropout here?
        self.attention = utils.QKVAttention(n_heads=num_heads)

        # Why do we have this
        self.conv2 = nn.Conv1d(channels, channels, 1)

    def forward(self, orig_batch):
        # Why don't we do silu here too?
        batch = self.norm1(orig_batch)

        assert(self.channels == batch.shape[1])
        batch = batch.reshape(batch.shape[0], self.channels, -1)

        qkv = self.conv1(batch)
        batch = self.attention(qkv)

        return batch.reshape(orig_batch.shape) + orig_batch
       
class Diffuser(torch.nn.Module):
    def __init__(self, device=None, channels=32, timestamp_channels=10, num_attentions=8, num_heads=1):
        super(Diffuser, self).__init__()
        # input is b x 3 x 28 x 28
        self.channels=channels
        self.timestamp_channels=timestamp_channels

        # work out math on this later
        # Wouldn't we want this to be a conv2d also?
        self.conv1 = nn.Conv1d(3+self.timestamp_channels, self.channels, 1) # why do we want more channels here?

        self.attention_resid_blocks = nn.ModuleList([])
        self.interpolate_blocks = nn.ModuleList([])
        self.dense_layers = nn.ModuleList([])

        print("Embedding dimensions:", channels)
        # With this list of items how does it magic the backwards pass?

        # attention residual blocks
        self.attention_resid_blocks.append(AttentionResid((28, 28), channels))
        self.attention_resid_blocks.append(AttentionResid((14, 14), channels*2))
        self.attention_resid_blocks.append(AttentionResid((7, 7), channels*4))
        self.attention_resid_blocks.append(AttentionResid((7, 7), channels*4))
        self.attention_resid_blocks.append(AttentionResid((14, 14), channels*2))
        self.attention_resid_blocks.append(AttentionResid((28, 28), channels))

        # downsampling blocks
        self.dense_layers.append(nn.Conv1d(channels, channels*2, 1))
        self.dense_layers.append(nn.Conv1d(channels*2, channels*4, 1))

        # upsampling blocks
        self.dense_layers.append(nn.Conv1d(channels*4, channels*2, 1))
        self.dense_layers.append(nn.Conv1d(channels*2, channels, 1))



        self.final_conv = nn.Conv1d(channels, 3, 1).to(device)

    def forward(self, steps, x):
        batch_len, init_channels, height, width = x.shape

        # sinusoidal timestep embedding
        timestamp_embedding = utils.timestep_embedding(steps, self.timestamp_channels)

        timestamp_embedding = torch.broadcast_to(timestamp_embedding.unsqueeze(2), (batch_len, self.timestamp_channels, height*width))

        data = x.reshape(batch_len, init_channels, -1)

        data_and_timestamps = torch.cat((data, timestamp_embedding), dim=1)

        y2 = F.silu(self.conv1(data_and_timestamps))
        
        intermediate = y2.reshape(batch_len, self.channels, height, width)

        # Do series of attention resid blocks
        intermediate = self.attention_resid_blocks[0](intermediate)
        intermediate = self.dense_layers[0](intermediate)
        intermediate = F.avg_pool2d(intermediate, (2, 2), 2)

        intermediate = self.attention_resid_blocks[1](intermediate)
        intermediate = self.dense_layers[1](intermediate)
        intermediate = F.avg_pool2d(intermediate, (2, 2), 2)

        intermediate = self.attention_resid_blocks[2](intermediate)
        intermediate = self.attention_resid_blocks[3](intermediate)

        intermediate = self.dense_layers[2](intermediate)
        intermediate = F.interpolate(intermediate, size=(14, 14), mode="nearest")
        intermediate = self.attention_resid_blocks[4](intermediate)

        intermediate = self.dense_layers[3](intermediate)
        intermediate = F.interpolate(intermediate, size=(28, 28), mode="nearest")
        intermediate = self.attention_resid_blocks[5](intermediate)

        final = self.final_conv(intermediate)

        return final.reshape(batch_len, 3, height, width)