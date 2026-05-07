from module.general_module import DepthwiseSeparableConv3d
from module.general_module import make_layer, ResidualBlock3D_NoBN, DepthwiseTransSeparableConv3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.general_module import ResBlock, make_layer
from module.sample_module import ModuleNet3D
from module.Quantization import Quantize_ste
from module.inv_module import D2DTInput
from module.general_module import SpaceTimePixelShuffle, SpaceTimePixelUnShuffle
from utils.registry import ARCH_REGISTRY

import os

class STREV_down(nn.Module):
    """
    空时下采样网络（STREV Downsample）。
    用于将高分辨率视频下采样到低分辨率。
    """
    def __init__(self, opt):
        super(STREV_down, self).__init__()
     
        self.mid_channels = opt['mid_channels']
        three_D_res_num = opt['three_D_res_num']
        kernel_size = opt['kernel_size']
        # 初始 3D 卷积层
        self.first_block = nn.Sequential(nn.Conv3d(3, self.mid_channels, 
                              kernel_size=kernel_size, 
                              padding=(1, (kernel_size[1]-1)//2, (kernel_size[2]-1)//2), 
                              stride=(1,1,1)),
                              nn.PReLU()) 
       
        # 3D 残差特征提取器
        self.feat_extractor = make_layer(ResidualBlock3D_NoBN, three_D_res_num, num_feat=self.mid_channels)
        # 采样头，用于调整到目标尺寸
        self.down_head = ModuleNet3D(self.mid_channels)

    def forward(self, imgs, target_size):
        feats = self.first_block(imgs)
        feats_3d = self.feat_extractor(feats)
        down_feat = self.down_head(feats_3d, target_size)
        return down_feat


class STREV_up(nn.Module):
    """
    空时上采样网络（STREV Upsample）。
    用于将低分辨率视频恢复到高分辨率。
    """
    def __init__(self, opt):
        super(STREV_up, self).__init__()
   
        self.mid_channels = opt['mid_channels']
        three_D_res_num = opt['three_D_res_num']
        # 初始 3D 卷积
        self.first_block = nn.Sequential(nn.Conv3d(opt['in_channels'], self.mid_channels//4, 
                              kernel_size=3, 
                              padding=(1, 1, 1), 
                              stride=(1,1,1)),
                              nn.PReLU()) 
        self.relu = nn.PReLU()
        # 密集连接的 3D 骨干网络
        self.dense3d_backbone = make_layer(D2DTInput, three_D_res_num, channel_in=opt['mid_channels'], channel_out=opt['mid_channels'])
        # 通道融合层
        self.fuse_channel_in = nn.Sequential(nn.Conv3d(opt['mid_channels'], opt['fuse_channels']*4, 
                              kernel_size=3, 
                              padding=(1, 1, 1), 
                              stride=(1,1,1)),
                              nn.PReLU()) 
        self.fuse_channel_out = nn.Sequential(nn.Conv3d(opt['mid_channels'], opt['fuse_channels']*4, 
                              kernel_size=3, 
                              padding=(1, 1, 1), 
                              stride=(1,1,1)),
                              nn.PReLU()) 
        # 采样头
        self.head = ModuleNet3D(in_channel=opt['fuse_channels'])
        # 空时像素重组/反重组
        self.STPS = SpaceTimePixelShuffle(r=1, s=2)
        self.UNSTPS = SpaceTimePixelUnShuffle(r=1, s=2)

    def forward(self, imgs, target_size):
        feats = self.first_block(imgs)
        feats = self.UNSTPS(feats) # 降低空间分辨率，增加通道数
        # 骨干网络特征提取与融合
        feats = self.fuse_channel_out(self.dense3d_backbone(feats)) + self.fuse_channel_in(feats)
        feats = self.STPS(feats) # 恢复空间分辨率
        x_up = self.head(feats, target_size)
        return x_up

@ARCH_REGISTRY.register()
class RescalerNet(nn.Module):
    """
    重缩放网络（RescalerNet），包含完整的下采样和上采样过程。
    """
    def __init__(self, opt):
        super(RescalerNet, self).__init__()
        down_opt = opt['down_opt']
        up_opt = opt['up_opt']
        self.quan_type = opt['quan_type']
        self.downet = STREV_down(down_opt)
        self.upnet = STREV_up(up_opt)
        # 量化层，用于模拟量化误差
        self.quan_layer = Quantize_ste(min_val=0.0, max_val=1.0)
      
    def forward(self, imgs, down_size, inference=False):
        """ 前向传播，包括下采样、量化（可选）和上采样。"""
        B, C, T, H, W = imgs.shape
        down_feat = self.downet(imgs, target_size=down_size)
        down_feat = torch.clamp(down_feat, 0, 1)
        if not inference:
            down_feat = self.quan_layer(down_feat)
        up_imgs = self.upnet(down_feat, target_size=(T, H, W)).contiguous()

        return down_feat, up_imgs

    @torch.no_grad()
    def inference_down(self, imgs, down_size):
        """ 仅执行下采样推理。"""
        out = self.downet(imgs, target_size=down_size)
        out = torch.clamp(out, 0, 1)
        return out

    @torch.no_grad()
    def inference_up(self, rev_back, size):
        """ 仅执行上采样推理。"""
        up_imgs = self.upnet(rev_back, target_size=size)
        up_imgs = torch.clamp(up_imgs, 0, 1)
        return up_imgs
