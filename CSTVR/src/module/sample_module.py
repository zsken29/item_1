import torch
import torch.nn as nn
import torch.nn.functional as F
from module.general_module import DepthwiseSeparableConv3d

def make_coord(shape, ranges=None, flatten=True):
    """ 在网格中心创建坐标。
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

class ModuleNet2d(nn.Module):
    """带输入卷积的残差块，用于 2D 图像缩放。
    """

    def __init__(self):
        super(ModuleNet2d, self).__init__()
        self.conv00 =  nn.Conv2d((64 + 2)*4+2, 256, 1)
        self.fc1 = nn.Conv2d(256, 256, 1)
        self.fc2 = nn.Conv2d(256, 3, 1)
        self.short_cut  = nn.Sequential(nn.Conv2d(64, 64, 1),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 3, 1)) 

    def query_rgb(self, feat, target_size):
        """ 查询目标尺寸下的 RGB 值（基于连续表示的思想）。"""
        feat_in = feat.clone()
        scale_max = 4
        H, W = target_size[0], target_size[1]
        scale_h = H / feat.shape[-2]
        scale_w = W / feat.shape[-1]
        coord = make_coord(target_size, flatten=False).cuda()
        coord = coord.unsqueeze(0).repeat(feat.shape[0], 1, 1, 1)

        cell = torch.ones(1, 2).cuda()
        cell[:, 0] *= 2 / H
        cell[:, 1] *= 2 / W
        cell_factor_h = max(scale_h / scale_max, 1)
        cell_factor_w = max(scale_w / scale_max, 1)
        cell[0][0] = cell[0][0] * cell_factor_h
        cell[0][1] = cell[0][1] * cell_factor_w

        pos_lr = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        rel_coords = []
        feat_s = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                feat_ = F.grid_sample(feat, coord_.flip(-1), mode='nearest', align_corners=False)
                old_coord = F.grid_sample(pos_lr, coord_.flip(-1), mode='nearest', align_corners=False)
                rel_coord = coord.permute(0, 3, 1, 2) - old_coord
                rel_coord[:, 0, :, :] *= feat.shape[-2]
                rel_coord[:, 1, :, :] *= feat.shape[-1]

                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)

                rel_coords.append(rel_coord)
                feat_s.append(feat_)
                
        rel_cell = cell.clone()
        rel_cell[:, 0] *= feat.shape[-2]
        rel_cell[:, 1] *= feat.shape[-1]

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t

        for index, area in enumerate(areas):
            feat_s[index] = feat_s[index] * (area / tot_area).unsqueeze(1)
    
        rel_cell = rel_cell.unsqueeze(-1).unsqueeze(-1).repeat(feat.shape[0], 1, coord.shape[1], coord.shape[2])
        grid = torch.cat([*rel_coords, *feat_s, rel_cell], dim=1)
       
        x = self.conv00(grid)
        ret = self.fc2(F.gelu(self.fc1(x)))
        
        short_cut = self.short_cut(feat_in)
        ret = ret + F.grid_sample(short_cut, coord.flip(-1), mode='bilinear', \
                                padding_mode='border', align_corners=False)
        return ret

    def forward(self, feat, size):
        return self.query_rgb(feat, size)

def make_coord_3d(shape, ranges=None, flatten=True, row_first=True):
    """ 在网格中心创建 3D 坐标。"""
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    if row_first:
        ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    else:
        ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='xy'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class ModuleNet3D(nn.Module):
    """ 用于 3D 视频缩放的模块。
    """

    def __init__(self, in_channel):
        super(ModuleNet3D, self).__init__()
        self.conv00_3d = DepthwiseSeparableConv3d(27 + in_channel * 8, 256, (1, 1, 1))
        self.fc1 = nn.Conv3d(256, 256, 1)
        self.fc2 = nn.Conv3d(256, 3, 1)
        self.short_cut = nn.Sequential(DepthwiseSeparableConv3d(in_channel, 3, (1, 1, 1))) 

    def query_rgb_3d(self, feat, target_size):
        """ 查询 3D 目标尺寸下的 RGB 值。"""
        in_b, in_c, in_t, in_h, in_w = feat.shape
        feat_in = feat.clone()
        T, H, W = target_size
        scale_t = T / feat.shape[-3]
        scale_h = H / feat.shape[-2]
        scale_w = W / feat.shape[-1]
        coord = make_coord_3d(target_size, flatten=False).cuda()
        coord = coord.unsqueeze(0).repeat(feat.shape[0], 1, 1, 1, 1)

        cell = torch.ones(1, 3).cuda()
        cell[:, 0] *= 2 / T
        cell[:, 1] *= 2 / H
        cell[:, 2] *= 2 / W
        cell_factor_t = scale_t 
        cell_factor_h = scale_h 
        cell_factor_w = scale_w
        cell[0, 0] = cell[0, 0] * cell_factor_t
        cell[0, 1] = cell[0, 1] * cell_factor_h
        cell[0, 2] = cell[0, 2] * cell_factor_w

        pos_lr = make_coord_3d(feat.shape[-3:], flatten=False).cuda() \
            .permute(3, 0, 1, 2) \
            .unsqueeze(0).expand(feat.shape[0], 3, *feat.shape[-3:])

        rt = 2 / feat.shape[-3] / 2
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        vt_lst = [-1, 1]
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        rel_coords = []
        feat_s = []
        areas = []
        for vt in vt_lst:
            for vx in vx_lst:
                for vy in vy_lst:
                    coord_ = coord.clone()
                    coord_[:, :, :, :, 0] += vt * rt + eps_shift
                    coord_[:, :, :, :, 1] += vx * rx + eps_shift
                    coord_[:, :, :, :, 2] += vy * ry + eps_shift
                    coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                    feat_ = F.grid_sample(feat, coord_.flip(-1), mode='nearest', align_corners=False)
                    old_coord = F.grid_sample(pos_lr, coord_.flip(-1), mode='nearest', align_corners=False)
                    rel_coord = coord.permute(0, 4, 1, 2, 3) - old_coord
                    rel_coord[:, 0, :, :, :] *= feat.shape[-3]
                    rel_coord[:, 1, :, :, :] *= feat.shape[-2]
                    rel_coord[:, 2, :, :, :] *= feat.shape[-1]

                    # 计算 3D 体积
                    area = torch.abs(rel_coord[:, 0, :, :, :] * rel_coord[:, 1, :, :, :] * rel_coord[:, 2, :, :, :])
                    areas.append(area + 1e-9)

                    rel_coords.append(rel_coord)
                    feat_s.append(feat_)
        
        rel_cell = cell.clone()
        rel_cell[:, 0] *= feat.shape[-3]  # T 维度缩放
        rel_cell[:, 1] *= feat.shape[-2]  # H 维度缩放
        rel_cell[:, 2] *= feat.shape[-1]  # W 维度缩放

        tot_volume = torch.stack(areas).sum(dim=0)

        # 调整体积顺序以进行 3D 加权
        t = areas[0]; areas[0] = areas[7]; areas[7] = t
        t = areas[1]; areas[1] = areas[6]; areas[6] = t
        t = areas[2]; areas[2] = areas[5]; areas[5] = t
        t = areas[3]; areas[3] = areas[4]; areas[4] = t

        for index, area in enumerate(areas):
            feat_s[index] = feat_s[index] * (area / tot_volume).unsqueeze(1)
       
            
        rel_cell = rel_cell.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(feat.shape[0], 1, coord.shape[1], coord.shape[2], coord.shape[3])
        grid = torch.cat([*rel_coords, *feat_s, rel_cell], dim=1)
      
        x = self.conv00_3d(grid)
        ret = self.fc2(F.gelu(self.fc1(x)))
        short_cut = self.short_cut(feat_in)
        short_cut = F.grid_sample(short_cut, coord.flip(-1), mode='bilinear', padding_mode='border', align_corners=False)
        ret = ret + short_cut
        return ret

    def forward(self, feat, size):
        return self.query_rgb_3d(feat, size)
 

 