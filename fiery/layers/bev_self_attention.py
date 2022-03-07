import torch
from self_attention_cv import MultiHeadSelfAttention
from torch import nn


class BEVSelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention_model = MultiHeadSelfAttention(dim=dim)

    def forward(self, bev_maps: torch.Tensor):
        """
        Args:
            bev_maps: [batch, num_cameras, C, H, W]
        """
        print(f'bev_maps: {bev_maps.shape}')
        bev_maps = bev_maps.flatten(2)

        print(f'bev_maps after reshape: {bev_maps.shape}')
        output = self.attention_model(bev_maps)

        print(f'output: {output.shape}')
        return output
