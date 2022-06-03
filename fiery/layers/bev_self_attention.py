import torch
from torch import nn


class BEVSelfAttention(nn.Module):
    def __init__(self, num_cameras, dim, num_heads=8):
        super().__init__()
        self.attention_model = nn.MultiheadAttention(dim, num_heads=num_heads)
        self.conv = nn.Conv2d(num_cameras * dim, dim, kernel_size=1)

    def forward(self, bev_maps: torch.Tensor):
        """
        Args:
            bev_maps: [batch, num_cameras, H, W, C]
        """
        # print(f'bev_maps: {bev_maps.shape}')
        B, num_cameras, H, W, C = bev_maps.shape
        # [B, n, H, W, C] -> [B*H*W, n, C]
        bev_maps = bev_maps.permute(0, 2, 3, 1, 4).flatten(0, 2)

        # print(f'bev_maps after reshape: {bev_maps.shape}')
        output = self.attention_model(query=bev_maps, key=bev_maps, value=bev_maps)

        # print(f'output: {output.shape}')
        # [B*H*W, n, C] -> [B, n*C, H, W]
        output = output.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        output = self.conv(output)
        # print(f'output after reshape: {output.shape}')
        return output
