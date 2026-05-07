
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def resize(x, scale_factor):
    """使用双线性插值调整张量大小。"""
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)

def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    """一个包含卷积层和 PReLU 激活层的简单序列模块。"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias), 
        nn.PReLU(out_channels)
    )

class ResidualBlock3D_NoBN(nn.Module):
    """不包含批归一化（BN）的 3D 残差块。

    结构如下:
        ---Conv-ReLU-Conv-+-
         |________________|

    参数:
        num_feat (int): 中间特征的通道数。默认值: 64。
        res_scale (float): 残差缩放比例。默认值: 1。
    """

    def __init__(self, num_feat=64, res_scale=1):
        super(ResidualBlock3D_NoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv3d(num_feat, num_feat,(3,3,3), 1, 1, bias=True)
        self.conv2 = nn.Conv3d(num_feat, num_feat, (3,3,3), 1, 1, bias=True)
        self.relu = nn.PReLU()


    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class DepthwiseSeparableConv3d(nn.Module):
    """3D 深度可分离卷积。"""
    def __init__(self, in_channels, out_channels,stride):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=stride,padding=1, groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class DepthwiseTransSeparableConv3d(nn.Module):
    """3D 深度可分离转置卷积（用于上采样）。"""
    def __init__(self, in_channels, out_channels,stride):
        super(DepthwiseTransSeparableConv3d, self).__init__()
        self.depthwise = nn.ConvTranspose3d(
        in_channels=in_channels,        
        out_channels=in_channels,       
        kernel_size=(3, 3, 3), 
        stride=stride,      
        padding=(1, 1, 1),     
        output_padding=(0, 1, 1) ,
        groups=in_channels
    )
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

    
    
class ResidualBlock3D_depthwise_NoBN(nn.Module):
    """使用深度可分离卷积的 3D 残差块。"""

    def __init__(self, num_feat=64, stride=1):
        super(ResidualBlock3D_depthwise_NoBN, self).__init__()
        self.res_scale = 1
        self.conv = DepthwiseSeparableConv3d(num_feat,num_feat,stride)
        self.relu = nn.PReLU()


    def forward(self, x):
        identity = x
        out = self.conv(x)
        return identity + out * self.res_scale


class ResBlock(nn.Module):
    """自定义残差块，支持部分通道的独立卷积。"""
    def __init__(self, in_channels, side_channels, bias=True):
        super(ResBlock, self).__init__()
        self.side_channels = side_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(side_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(in_channels)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(side_channels)
        )
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        out = self.conv1(x)
        # 对部分通道进行额外的卷积处理
        out[:, -self.side_channels:, :, :] = self.conv2(out[:, -self.side_channels:, :, :].clone())
        out = self.conv3(out)
        out[:, -self.side_channels:, :, :] = self.conv4(out[:, -self.side_channels:, :, :].clone())
        out = self.prelu(x + self.conv5(out))
        return out


class Encoder(nn.Module):
    """金字塔结构编码器，用于提取多尺度特征。"""
    def __init__(self,num_feats = [24,36,54,72]):
        super(Encoder, self).__init__()
        self.pyramid1 = nn.Sequential(
            convrelu(3, num_feats[0], 3, 2, 1), 
            convrelu(num_feats[0], num_feats[0], 3, 1, 1)
        )
        self.pyramid2 = nn.Sequential(
            convrelu(num_feats[0], num_feats[1], 3, 2, 1), 
            convrelu(num_feats[1], num_feats[1], 3, 1, 1)
        )
        self.pyramid3 = nn.Sequential(
            convrelu(num_feats[1], num_feats[2], 3, 2, 1), 
            convrelu(num_feats[2], num_feats[2], 3, 1, 1)
        )
      
        
    def forward(self, img):
        f1 = self.pyramid1(img)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        return f1, f2, f3



class ResidualBlockNoBN(nn.Module):
    """不包含批归一化（BN）的 2D 残差块。"""

    def __init__(self, mid_channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

def make_layer(basic_block, num_basic_block, **kwarg):
    """通过堆叠相同的块来创建层。

    参数:
        basic_block (nn.module): 基础块的类。
        num_basic_block (int): 块的数量。

    返回:
        nn.Sequential: 堆叠后的 nn.Sequential 模块。
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

class ResidualBlocksWithInputConv(nn.Module):
    """前面带有一个卷积层的残差块序列。
    参数:
        in_channels (int): 第一个卷积层的输入通道数。
        out_channels (int): 残差块的通道数。默认值: 64。
        num_blocks (int): 残差块的数量。默认值: 30。
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # 匹配残差块通道数的卷积层
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # 堆叠残差块
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        return self.main(feat)


class SpaceTimePixelShuffle(nn.Module):
    """空时像素重组层，用于同时在空间和时间维度上进行上采样。"""
    def __init__(self, r, s):
        super(SpaceTimePixelShuffle, self).__init__()
        self.r = r # 时间缩放因子
        self.s = s # 空间缩放因子

    def forward(self, x):
        b, c, t, h, w = x.size()
        # 通道数必须能被 r * s * s 整除
        out_c = c // (self.r * self.s * self.s)
        x = x.view(b, out_c, self.r, self.s, self.s, t, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return x.view(b, out_c, t * self.r, h * self.s, w * self.s)

class SpaceTimePixelUnShuffle(nn.Module):
    """空时像素反重组层，用于同时在空间和时间维度上进行下采样。"""
    def __init__(self, r, s):
        super(SpaceTimePixelUnShuffle, self).__init__()
        self.r = r # 时间缩放因子
        self.s = s # 空间缩放因子

    def forward(self, x):
        b, c, t, h, w = x.size()
        out_c = c * (self.r * self.s * self.s)
        x = x.view(b, c, t // self.r, self.r, h // self.s, self.s, w // self.s, self.s)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        return x.view(b, out_c, t // self.r, h // self.s, w // self.s)
        self.r = r  # Time upscaling factor
        self.s = s  # Spatial upscaling factor

    def forward(self, x):
        b, c, t, h, w = x.size()
        c_out = int( c / (self.r * self.s * self.s))

        # First, reshape to split the spatial and temporal dimensions
        x = x.view(b, c_out, self.r, self.s, self.s, t, h, w)
        
        # Next, permute to interleave the spatial and temporal dimensions 这个应该是对的
        # (b, c_out, self.r, self.s, self.s, t, h, w) -> (b, c_out, t,self.r,h,self.s, w,self.s)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        # Finally, reshape to the desired output shape
        x = x.view(b, c_out, self.r * t, self.s * h, self.s * w)
        return x
class SpaceTimePixelUnShuffle(nn.Module):
    def __init__(self, r, s):
        super(SpaceTimePixelUnShuffle, self).__init__()
        self.r = r  # Time downscaling factor
        self.s = s  # Spatial downscaling factor

    def forward(self, x):
        b, c, t, h, w = x.size()
        c_out = c * (self.r * self.s * self.s)

        # First, reshape to prepare for permutation
        x = x.view(b, c, t // self.r, self.r, h // self.s, self.s, w // self.s, self.s)

        # Next, permute to rearrange the spatial and temporal dimensions
        # (b, c, t ,r, h, s, w, s) ->(b,c,r,s,s,t,h,w)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()

        # Finally, reshape to the desired output shape
        x = x.view(b, c_out, t // self.r, h // self.s, w // self.s)

        return x
# Example usage:


if __name__=='__main__':
    pass



            
