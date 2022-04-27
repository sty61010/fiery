from typing import List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FrustumToBEV(nn.Module):
    def __init__(self, grid_size: List[int], pc_range: List[int], d_bound: List[int]):
        """Initializes Grid Generator for frustum features

        Args:
            grid_size (list): Voxel shape [X, Y, Z]
            pc_range (list): Voxelization point cloud range [X_min, Y_min, Z_min, X_max, Y_max, Z_max]
            d_bound (list): Depth bound [depth_start, depth_end, depth_step]
        """
        super().__init__()
        self.grid_size = torch.tensor(grid_size)
        pc_range = torch.tensor(pc_range).reshape(2, 3)
        self.pc_min = pc_range[0]
        self.pc_max = pc_range[1]
        self.d_bound = d_bound

        num_xyz_points = ((self.pc_max - self.pc_min) // self.grid_size).long()
        x, y, z = [torch.linspace(pc_min, pc_max, num_points)
                   for pc_min, pc_max, num_points, size in zip(self.pc_min, self.pc_max, num_xyz_points, self.grid_size)]
        # [X, Y, Z, 3]
        self.grid = torch.stack(torch.meshgrid(x, y, z), dim=-1)
        # [X, Y, Z, 4]
        self.grid = torch.cat([self.grid, torch.ones((*self.grid.shape[:3], 1))], dim=-1)

    def forward(
        self,
        frustum: torch.Tensor,
        intrinsics: torch.Tensor,
        lidar_to_sensor: torch.Tensor,
        img_shape: List[int]
    ) -> torch.Tensor:
        """Generate BEV map from front-view cameras

        Args:
            frustum (Tensor): [B, num_cameras, D, H, W, C]
            intrinsics (Tensor): [B, num_cameras, 3, 3]
            lidar_to_sensor (Tensor): [B, num_cameras, 4, 4] or [B, num_cameras, 3, 4]
            img_shape (list): Image shape [height, width].
                Note that this is not the input shape of the frustum.
                This is the shape with respect to the intrinsic.
        Returns:
            bev_map (Tensor): [B, C, Y, X]
        """
        assert frustum.shape[:2] == intrinsics.shape[:2] == lidar_to_sensor.shape[:2]
        batch, num_cameras, _, _ = intrinsics.shape
        # [B*num_cameras, C, D, H, W]
        frustum = frustum.flatten(0, 1).permute(0, 4, 1, 2, 3)
        _, _, D, _, _ = frustum.shape

        # [B, num_cameras, 3, 4]
        proj = (intrinsics @ lidar_to_sensor[:, :, :3])
        # [B, num_cameras, X, Y, Z, 3]
        uvd = torch.einsum('bnij,xyzj->bnxyzi', proj, self.grid.to(proj.device))
        # convert from homogeneous coordinate to image coordinate
        uv, depth = uvd[..., :2] / uvd[..., -1:], uvd[..., -1:]
        depth_bin = (depth - self.d_bound[0]) / self.d_bound[2]
        # [B, num_cameras, X, Y, Z, 3]
        uvd = torch.cat([uv, depth_bin], dim=-1)

        # [B * num_cameras, X, Y, Z, 3]
        uvd = uvd.flatten(0, 1)
        # normalize to [-1, 1]
        img_H, img_W = img_shape
        uvd = 2 * uvd / torch.tensor([img_W, img_H, D], dtype=uvd.dtype, device=uvd.device).reshape(1, 1, 1, 1, 3) - 1

        # [B*num_cameras, C, X, Y, Z]
        voxel = F.grid_sample(frustum, uvd, align_corners=False)

        # [B, num_cameras, C, X, Y, Z]
        voxel = voxel.reshape((batch, num_cameras, *voxel.shape[1:]))
        # [B, C, Y, X]
        bev = voxel.permute(0, 1, 5, 2, 4, 3).sum(dim=(1, 2))
        return bev


if __name__ == '__main__':
    """
    How to run the test:
        python -m fiery.layers.frustum_to_bev --config fiery/configs/literature/lss_mini.yml
    """
    from ..config import get_cfg, get_parser
    from ..data import prepare_dataloaders
    from torchvision.utils import save_image

    def hsv_to_rgb(h, s, v):
        if torch.all(s == 0.0):
            return v, v, v
        shape = v.shape
        h, s, v = h.flatten(), s.flatten(), v.flatten()
        i = (h * 6.0).long()
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        combination = [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]]
        # [N, 6, 3]
        rgb = torch.stack([torch.stack(com, dim=1) for com in combination], dim=1)
        rgb = rgb[torch.arange(rgb.shape[0]), i].reshape(*shape, 3)
        return rgb[..., 0], rgb[..., 1], rgb[..., 2]

    args = get_parser().parse_args()
    cfg = get_cfg(args)

    torch.set_printoptions(sci_mode=False)
    height, width = int(cfg.IMAGE.FINAL_DIM[0] // 8), int(cfg.IMAGE.FINAL_DIM[1] // 8)

    frustum = torch.zeros((cfg.IMAGE.N_CAMERA * width, 3))
    frustum[..., 0] = torch.linspace(0., 1.0, cfg.IMAGE.N_CAMERA * width)
    frustum[..., 1] = 0.2
    frustum[..., 2] = 0.2
    frustum = frustum.reshape(cfg.IMAGE.N_CAMERA, width, 3).repeat(1, height, 1).reshape(cfg.IMAGE.N_CAMERA, height, width, 3).permute(0, 3, 1, 2)
    save_image(torch.stack(hsv_to_rgb(frustum[:, 0], frustum[:, 1], frustum[:, 2]), dim=1), 'frustum_2d.png')
    frustum = torch.einsum('nchw,dc->ndhwc', frustum, torch.stack([torch.ones(48), torch.ones(48), torch.linspace(1, 0.2, 48)], dim=1))
    frustum = torch.stack(hsv_to_rgb(frustum[..., 0], frustum[..., 1], frustum[..., 2]), dim=-1)
    save_image(frustum.permute(0, 1, 4, 2, 3).flatten(0, 1), 'frustum.png')
    frustum = torch.stack([frustum for _ in range(cfg.VAL_BATCHSIZE)], dim=0)
    print(f'frustum: {frustum.shape}')

    trainloader, valloader, testloader = prepare_dataloaders(cfg)
    batch = next(iter(valloader))

    intrinsics = batch.get('intrinsics')
    extrinsics = batch.get('extrinsics')
    print(f'intrinsics: {intrinsics.shape}')
    intrinsics = intrinsics.flatten(0, 1)
    extrinsics = extrinsics.flatten(0, 1)

    model = FrustumToBEV(np.array([0.5, 0.5, 1]), [-50., -50., -3, 50., 50., 5.], cfg.LIFT.D_BOUND)
    bev = model(frustum, intrinsics, extrinsics, cfg.IMAGE.FINAL_DIM)
    bev = bev.clamp(0., 1.)
    save_image(bev, 'bev.png', pad_value=1.)
