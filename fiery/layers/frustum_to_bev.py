
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FrustumToBEV(nn.Module):
    def __init__(self, grid_size, pc_range, d_bound):
        """Initializes Grid Generator for frustum features

        Args:
            grid_size [np.array(3)]: Voxel grid shape [X, Y, Z]
            pc_range [list]: Voxelization point cloud range [X_min, Y_min, Z_min, X_max, Y_max, Z_max]
        """
        super().__init__()
        self.grid_size = torch.tensor(grid_size)
        pc_range = torch.tensor(pc_range).reshape(2, 3)
        self.pc_min = pc_range[0]
        self.pc_max = pc_range[1]
        self.d_bound = d_bound

        num_xyz_points = ((self.pc_max - self.pc_min) // self.grid_size).long()
        x, y, z = [torch.linspace(pc_min, pc_max - size, num_points) + size / 2
                   for pc_min, pc_max, num_points, size in zip(self.pc_min, self.pc_max, num_xyz_points, self.grid_size)]
        # x, y, z = [torch.arange(pc_min, pc_max, size) + size / 2
        #            for pc_min, pc_max, size in zip(self.pc_min, self.pc_max, self.grid_size)]

        # [X, Y, Z, 3]
        self.grid = torch.stack(torch.meshgrid(x, y, z), dim=-1)
        # [X, Y, Z, 4]
        self.grid = torch.cat([self.grid, torch.ones((*self.grid.shape[:3], 1))], dim=-1)
        print(f'grid: {self.grid}')

    def forward(self, frustum, intrinsics, lidar_to_sensor):
        """
        Args:
            frustum: [B, num_cameras, D, H, W, C]
            intrinsics: [B, num_cameras, 3, 3]
            lidar_to_sensor: [B, num_cameras, 4, 4] or [B, num_cameras, 3, 4]

        Returns:
            bev_map: [B, C, Y, X]
        """
        batch, num_cameras, _, _ = intrinsics.shape
        frustum = frustum.flatten(0, 1).permute(0, 4, 1, 2, 3)
        _, _, D, H, W = frustum.shape

        # [B, num_cameras, X, Y, Z, 3]
        grid_in_sensor_coord = torch.linalg.solve(lidar_to_sensor.unsqueeze(2).unsqueeze(2).unsqueeze(2), self.grid.to(frustum.device).unsqueeze(0).unsqueeze(0).unsqueeze(-1)).squeeze(-1)[..., :3]
        # grid_in_sensor_coord = torch.einsum('bnij,...j->bn...i', lidar_to_sensor, self.grid.to(frustum.device))[..., :3]
        # print(f'gisc: {grid_in_sensor_coord}')
        # [B, num_cameras, X, Y, Z, 3]
        uvd = (intrinsics.unsqueeze(2).unsqueeze(2).unsqueeze(2) @ grid_in_sensor_coord.unsqueeze(-1)).squeeze(-1)
        # uvd = torch.einsum('bnij,bnxyzj->bnxyzi', intrinsics, grid_in_sensor_coord)
        uv, depth = uvd[..., :2] / uvd[..., -1:], uvd[..., -1:]
        depth_bin = (depth - self.d_bound[0]) / self.d_bound[2]
        # [B, num_cameras, X, Y, Z, 3]
        uvd = torch.cat([uv, depth_bin], dim=-1)
        # print(uvd)

        # [B * num_cameras, X, Y, Z, 3]
        uvd = uvd.flatten(0, 1)
        # normalize to [-1, 1]
        uvd = 2 * uvd / torch.tensor([W, H, D], dtype=uvd.dtype, device=uvd.device).reshape(1, 1, 1, 1, 3) - 1

        # [B*num_cameras, C, X, Y, Z]
        voxel = F.grid_sample(frustum, uvd, align_corners=True)
        # [B, num_cameras, C, X, Y, Z]
        voxel = voxel.reshape((batch, num_cameras, *voxel.shape[1:]))
        # [B, C, Y, X]
        # bev = voxel.permute(0, 1, 5, 2, 4, 3).flatten(0, 2)
        # bev = voxel.sum(dim=1).permute(0, 4, 1, 3, 2).flatten(0, 1)
        bev = voxel.sum(dim=(1, 5)).permute(0, 1, 3, 2)
        return bev


if __name__ == '__main__':
    from torchvision.utils import save_image
    from fiery.config import get_parser, get_cfg
    from fiery.data import prepare_dataloaders

    def hsv_to_rgb(h, s, v):
        if torch.all(s == 0.0):
            return v, v, v
        shape = v.shape
        h, s, v = h.flatten(), s.flatten(), v.flatten()
        i = (h * 6.0).long()  # XXX assume int() truncates!
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

    frustum = torch.zeros((cfg.IMAGE.N_CAMERA * 88, 3))
    frustum[..., 0] = torch.linspace(0., 1.0, cfg.IMAGE.N_CAMERA * 88)
    frustum[..., 1:] = 0.5
    frustum = frustum.reshape(cfg.IMAGE.N_CAMERA, 88, 3).repeat(1, 32, 1).reshape(cfg.IMAGE.N_CAMERA, 32, 88, 3).permute(0, 3, 1, 2)
    save_image(torch.stack(hsv_to_rgb(frustum[:, 0], frustum[:, 1], frustum[:, 2]), dim=1), 'frustum_2d.png')
    frustum = torch.einsum('nchw,dc->ndhwc', frustum, torch.stack([torch.ones(48), torch.linspace(1, 0.2, 48), torch.ones(48)], dim=1))
    frustum = torch.stack(hsv_to_rgb(frustum[..., 0], frustum[..., 1], frustum[..., 2]), dim=-1)
    print(frustum.shape)
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
    # extrinsics = torch.linalg.inv(extrinsics.flatten(0, 1))

    model = FrustumToBEV(np.array([0.25, 0.25, 1]), [-50., -50., -3, 50., 50., 5.], cfg.LIFT.D_BOUND)
    # model = FrustumToBEV(np.array([2, 2, 8]), [-5., -5., -3., 5., 5., 5.], cfg.LIFT.D_BOUND)
    bev = model(frustum, intrinsics, extrinsics)
    bev = bev.clamp(0., 1.)
    save_image(bev, 'bev.png', pad_value=1.)
