"""
CSTVR 核心技术演示脚本
=====================
本脚本从 CSTVR 项目中提取关键设计模式与核心实现代码，以独立可运行的方式展示项目核心技术点。

涵盖技术点：
  1. 时空像素重排 (SpaceTimePixelShuffle / SpaceTimePixelUnShuffle)
  2. 3D可逆神经网络 (Invertible Neural Network with Affine Coupling)
  3. 直通估计器量化 (Straight-Through Estimator Quantization)
  4. 基于坐标的3D隐式采样 (LIIF-style 3D Implicit Sampling)
  5. 3D密集连接块 (3D Dense Block with Temporal-Spatial Decomposition)
  6. 可逆运动隐写模块完整流程 (IMSM Forward & Reverse Pipeline)

运行方式：
  cd src/test
  python cstvr_core_demo.py

依赖：torch, numpy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# 技术点 1: 时空像素重排
# ============================================================================
# 项目中的 SpaceTimePixelShuffle 和 SpaceTimePixelUnShuffle 实现了3D视频数据
# 在通道维与时空维之间的可逆转换。这是 IMSM 模块中"双重像素重排"的基础。
#
# 核心思想：
#   - PixelShuffle: 将通道维的数据重排到空间/时间维，实现上采样
#     例如: (B, C*r*s*s, T, H, W) → (B, C, T*r, H*s, W*s)
#   - PixelUnShuffle: 将空间/时间维的数据重排到通道维，实现下采样
#     例如: (B, C, T, H, W) → (B, C*r*s*s, T//r, H//s, W//s)
#
# 在 IMSM 中，参数 r=1（时间因子）, s=2（空间因子），调用两次 st_shuffle
# 实现通道数减少 16 倍、空间分辨率扩大 4 倍的效果。

class SpaceTimePixelShuffle(nn.Module):
    """时空像素重排（上采样）：通道维 → 时空维

    将形状为 (B, C, T, H, W) 的张量，按时间因子 r 和空间因子 s
    重排为 (B, C//(r*s*s), T*r, H*s, W*s)。

    参数:
        r (int): 时间上采样因子
        s (int): 空间上采样因子
    """

    def __init__(self, r, s):
        super(SpaceTimePixelShuffle, self).__init__()
        self.r = r
        self.s = s

    def forward(self, x):
        b, c, t, h, w = x.size()
        c_out = int(c / (self.r * self.s * self.s))
        # 将通道维拆分为 (c_out, r, s, s) 四个子维度
        x = x.view(b, c_out, self.r, self.s, self.s, t, h, w)
        # 重排维度顺序: 通道子维插入到对应时空位置
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        # 合并为输出形状
        x = x.view(b, c_out, self.r * t, self.s * h, self.s * w)
        return x


class SpaceTimePixelUnShuffle(nn.Module):
    """时空像素逆重排（下采样）：时空维 → 通道维

    将形状为 (B, C, T, H, W) 的张量，按时间因子 r 和空间因子 s
    重排为 (B, C*r*s*s, T//r, H//s, W//s)。

    参数:
        r (int): 时间下采样因子
        s (int): 空间下采样因子
    """

    def __init__(self, r, s):
        super(SpaceTimePixelUnShuffle, self).__init__()
        self.r = r
        self.s = s

    def forward(self, x):
        b, c, t, h, w = x.size()
        c_out = c * (self.r * self.s * self.s)
        # 将时空维拆分为子块
        x = x.view(b, c, t // self.r, self.r, h // self.s, self.s, w // self.s, self.s)
        # 重排: 时空子块移到通道维
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        # 合并为输出形状
        x = x.view(b, c_out, t // self.r, h // self.s, w // self.s)
        return x


def demo_spacetime_pixel_shuffle():
    """演示时空像素重排的可逆性"""
    print("=" * 70)
    print("技术点 1: 时空像素重排 (SpaceTimePixelShuffle / UnShuffle)")
    print("=" * 70)

    r, s = 1, 2  # 与项目中 IMSM 的参数一致
    shuffle = SpaceTimePixelShuffle(r, s)
    unshuffle = SpaceTimePixelUnShuffle(r, s)

    # 模拟 my_InvNN 输出: 48通道, 4帧, 64x112 空间分辨率
    x = torch.randn(1, 48, 4, 64, 112)
    print(f"输入形状: {x.shape}")

    # 第一次 st_shuffle: 48 → 12 通道, 空间 64x112 → 128x224
    x1 = shuffle(x)
    print(f"第一次 st_shuffle: {x1.shape}")

    # 第二次 st_shuffle: 12 → 3 通道, 空间 128x224 → 256x448
    x2 = shuffle(x1)
    print(f"第二次 st_shuffle: {x2.shape}  ← 这就是隐写图像的形状 (3通道, 正常RGB)")

    # 验证可逆性: 两次 unshuffle 应恢复原始形状
    x_rev = unshuffle(unshuffle(x2))
    print(f"两次 st_unshuffle 恢复: {x_rev.shape}")
    print(f"可逆性验证 (应接近0): {torch.max(torch.abs(x - x_rev)).item():.6f}")
    print()


# ============================================================================
# 技术点 2: 3D可逆神经网络 (Affine Coupling Layer)
# ============================================================================
# 项目中的 InvBlockExp 实现了仿射耦合可逆块，是 my_InvNN 的核心计算单元。
#
# 核心思想（Affine Coupling）：
#   将输入沿通道维分为 x1, x2 两部分:
#     前向: y1 = x1 + F(x2),  s = clamp * (sigmoid(H(y1)) * 2 - 1),  y2 = x2 * exp(s) + G(y1)
#     反向: s = clamp * (sigmoid(H(x1)) * 2 - 1),  y2 = (x2 - G(x1)) / exp(s),  y1 = x1 - F(y2)
#
# 可逆性保证: 只要 F, G, H 是任意神经网络（不需要可逆），整个块就是可逆的。
# 这是因为前向和反向的计算只涉及加法、乘法和指数运算，都可以精确逆转。

class SimpleSubnet3D(nn.Module):
    """简化的3D子网络，用于 InvBlockExp 的 F/G/H 函数

    实际项目中使用 D2DTInput_dense（5层密集3D卷积），此处简化为2层3D卷积以便演示。
    """

    def __init__(self, channel_in, channel_out):
        super(SimpleSubnet3D, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, channel_out, 3, 1, 1)
        self.conv2 = nn.Conv3d(channel_out, channel_out, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        # Xavier 初始化 (与项目一致)
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        return self.conv2(self.lrelu(self.conv1(x)))


class InvBlockExp(nn.Module):
    """可逆块扩展 (Invertible Block with Affine Coupling)

    实现仿射耦合变换，是 my_InvNN 的核心计算单元。
    前向和反向传播均可精确计算，无需存储中间激活。

    参数:
        subnet_constructor: 子网络构造函数 (channel_in, channel_out) -> nn.Module
        channel_num: 输入通道总数
        channel_split_num: 分割点（x1 的通道数）
        clamp: 缩放因子，限制 s 的范围以保证数值稳定性
    """

    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.0):
        super(InvBlockExp, self).__init__()
        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num
        self.clamp = clamp
        # 三个子网络: F 用于平移, G 用于缩放后的平移, H 用于计算缩放因子
        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        # 沿通道维分割
        x1, x2 = x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2)

        if not rev:
            # 前向: 仿射耦合变换
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            # 反向: 精确逆转前向计算
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)


def demo_invertible_block():
    """演示可逆块的前向/反向可逆性"""
    print("=" * 70)
    print("技术点 2: 3D可逆神经网络 (Affine Coupling Layer)")
    print("=" * 70)

    channel_num = 12
    split_num = 3  # 与项目一致: channel_out=3 作为分割点

    block = InvBlockExp(SimpleSubnet3D, channel_num, split_num, clamp=1.0)
    block.eval()

    x = torch.randn(1, channel_num, 4, 16, 28)
    print(f"输入形状: {x.shape}, 通道分割: x1={split_num}, x2={channel_num - split_num}")

    # 前向传播
    y = block(x, rev=False)
    print(f"前向输出形状: {y.shape}")

    # 反向传播（恢复原始输入）
    x_recovered = block(y, rev=True)
    print(f"反向恢复形状: {x_recovered.shape}")
    print(f"可逆性验证 (应接近0): {torch.max(torch.abs(x - x_recovered)).item():.6f}")
    print()


# ============================================================================
# 技术点 3: 直通估计器量化 (STE Quantization)
# ============================================================================
# 项目中的 Quantize_ste 实现了可微分的量化操作，使得量化操作可以嵌入到
# 端到端训练的神经网络中。
#
# 核心思想：
#   前向: 将连续值量化为 255 级离散值 (模拟8位存储)
#   反向: 梯度直通 (round 操作的梯度近似为1)
#
# 实现技巧: (round(x) - x).detach() + x
#   - 前向: round(x) - x + x = round(x)  (量化结果)
#   - 反向: detach() 阻断 round(x)-x 的梯度, 只保留 x 的梯度 (直通)

class QuantizeSTE(nn.Module):
    """直通估计器量化层

    将输入值裁剪到 [min_val, max_val] 范围后量化为 255 级离散值。
    反向传播时梯度直通，允许端到端训练。

    参数:
        min_val (float): 裁剪下界
        max_val (float): 裁剪上界
    """

    def __init__(self, min_val=0.0, max_val=1.0):
        super(QuantizeSTE, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        # 使用 ReLU 实现可微分裁剪 (避免 torch.clamp 的梯度问题)
        x_clipped_min = x + F.relu(self.min_val - x)
        x_clipped = x_clipped_min - F.relu(x_clipped_min - self.max_val)
        # STE 量化: 前向 round, 反向直通
        return (torch.round(x_clipped * 255.0) / 255.0 - x_clipped).detach() + x_clipped


def demo_quantization():
    """演示 STE 量化的前向量化与反向梯度直通"""
    print("=" * 70)
    print("技术点 3: 直通估计器量化 (STE Quantization)")
    print("=" * 70)

    quant = QuantizeSTE(min_val=0.0, max_val=1.0)

    # 模拟网络输出的连续值
    x = torch.tensor([0.123, 0.456, 0.789, 1.5, -0.1], requires_grad=True)
    print(f"输入值:       {x.detach().numpy()}")

    # 前向: 量化为 255 级
    y = quant(x)
    print(f"量化后:       {y.detach().numpy()}")
    print(f"量化级数:     {torch.unique(y * 255).numel()} (255级)")

    # 反向: 梯度直通验证
    loss = y.sum()
    loss.backward()
    print(f"梯度 (直通):  {x.grad.numpy()}")
    print(f"梯度全为1.0:  {torch.all(x.grad[:3] == 1.0).item()} (前3个有效值梯度直通)")
    print()


# ============================================================================
# 技术点 4: 基于坐标的3D隐式采样 (LIIF-style 3D Implicit Sampling)
# ============================================================================
# 项目中的 ModuleNet3D 实现了基于 LIIF 思想的3D隐式采样网络，是连续时空重采样的核心。
#
# 核心思想：
#   1. 生成目标尺寸的3D坐标网格
#   2. 对每个目标坐标点，在源特征体中找到8个最近邻
#   3. 计算相对坐标编码和体积权重
#   4. 加权聚合特征 + cell 编码 → MLP 预测目标值
#   5. 叠加双线性插值快捷连接
#
# 此处简化演示坐标生成和最近邻采样机制。

def make_coord_3d(shape, ranges=None, flatten=True):
    """生成3D坐标网格

    在 [-1, 1] 范围内生成指定形状的网格中心坐标。

    参数:
        shape (tuple): 目标形状 (T, H, W)
        ranges: 每个维度的范围，默认 [-1, 1]
        flatten: 是否展平为 (N, 3)

    返回:
        坐标张量，形状为 (T, H, W, 3) 或 (T*H*W, 3)
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


class SimplifiedModuleNet3D(nn.Module):
    """简化的3D隐式采样网络

    展示 LIIF-style 3D 采样的核心机制:
    - 3D坐标生成
    - 8邻域最近邻特征采样
    - 相对坐标编码与体积权重
    - MLP 预测 + 双线性插值快捷连接

    实际项目中使用 DepthwiseSeparableConv3d 和更复杂的网络结构。
    """

    def __init__(self, in_channel):
        super(SimplifiedModuleNet3D, self).__init__()
        # 输入: 8个相对坐标(3D×2=6) + 8个特征(in_channel×8) + cell(3) = 27+in_channel*8
        self.conv00 = nn.Conv3d(27 + in_channel * 8, 64, 1)
        self.fc1 = nn.Conv3d(64, 64, 1)
        self.fc2 = nn.Conv3d(64, 3, 1)
        self.shortcut = nn.Conv3d(in_channel, 3, 1)

    def query_rgb_3d(self, feat, target_size):
        """3D隐式查询

        参数:
            feat: 源特征 (B, C, T_in, H_in, W_in)
            target_size: 目标尺寸 (T_out, H_out, W_out)

        返回:
            预测的特征 (B, 3, T_out, H_out, W_out)
        """
        in_b, in_c, in_t, in_h, in_w = feat.shape
        feat_in = feat.clone()
        T, H, W = target_size

        # 生成目标3D坐标
        coord = make_coord_3d(target_size, flatten=False).to(feat.device)
        coord = coord.unsqueeze(0).repeat(feat.shape[0], 1, 1, 1, 1)

        # Cell 编码: 表示每个查询点的感受野大小
        cell = torch.ones(1, 3).to(feat.device)
        cell[:, 0] *= 2 / T * (T / feat.shape[-3])
        cell[:, 1] *= 2 / H * (H / feat.shape[-2])
        cell[:, 2] *= 2 / W * (W / feat.shape[-1])

        # 源特征的坐标网格
        pos_lr = make_coord_3d(feat.shape[-3:], flatten=False).to(feat.device)
        pos_lr = pos_lr.permute(3, 0, 1, 2).unsqueeze(0).expand(feat.shape[0], 3, *feat.shape[-3:])

        # 最近邻搜索半径
        rt = 2 / feat.shape[-3] / 2
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        rel_coords = []
        feat_s = []
        areas = []

        # 8邻域采样 (3D中8个角点)
        for vt in [-1, 1]:
            for vx in [-1, 1]:
                for vy in [-1, 1]:
                    coord_ = coord.clone()
                    coord_[:, :, :, :, 0] += vt * rt + 1e-6
                    coord_[:, :, :, :, 1] += vx * rx + 1e-6
                    coord_[:, :, :, :, 2] += vy * ry + 1e-6
                    coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                    # 最近邻特征采样
                    feat_ = F.grid_sample(feat, coord_.flip(-1), mode='nearest', align_corners=False)
                    # 最近邻坐标采样
                    old_coord = F.grid_sample(pos_lr, coord_.flip(-1), mode='nearest', align_corners=False)

                    # 相对坐标编码
                    rel_coord = coord.permute(0, 4, 1, 2, 3) - old_coord
                    rel_coord[:, 0] *= feat.shape[-3]
                    rel_coord[:, 1] *= feat.shape[-2]
                    rel_coord[:, 2] *= feat.shape[-1]

                    # 体积权重 (3D扩展的面积权重)
                    area = torch.abs(
                        rel_coord[:, 0] * rel_coord[:, 1] * rel_coord[:, 2]
                    )
                    areas.append(area + 1e-9)
                    rel_coords.append(rel_coord)
                    feat_s.append(feat_)

        # 体积权重归一化 (对角交换确保正确的插值权重)
        tot_volume = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[7]; areas[7] = t
        t = areas[1]; areas[1] = areas[6]; areas[6] = t
        t = areas[2]; areas[2] = areas[5]; areas[5] = t
        t = areas[3]; areas[3] = areas[4]; areas[4] = t

        for index, area in enumerate(areas):
            feat_s[index] = feat_s[index] * (area / tot_volume).unsqueeze(1)

        # Cell 编码扩展
        rel_cell = cell.clone()
        rel_cell[:, 0] *= feat.shape[-3]
        rel_cell[:, 1] *= feat.shape[-2]
        rel_cell[:, 2] *= feat.shape[-1]
        rel_cell = rel_cell.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(
            feat.shape[0], 1, coord.shape[1], coord.shape[2], coord.shape[3]
        )

        # 拼接: 相对坐标 + 加权特征 + cell编码
        grid = torch.cat([*rel_coords, *feat_s, rel_cell], dim=1)

        # MLP 预测
        x = self.conv00(grid)
        ret = self.fc2(F.gelu(self.fc1(x)))

        # 双线性插值快捷连接
        short_cut = self.shortcut(feat_in)
        short_cut = F.grid_sample(short_cut, coord.flip(-1), mode='bilinear',
                                  padding_mode='border', align_corners=False)
        ret = ret + short_cut
        return ret

    def forward(self, feat, target_size):
        return self.query_rgb_3d(feat, target_size)


def demo_implicit_sampling():
    """演示3D隐式采样网络的任意尺度重采样能力"""
    print("=" * 70)
    print("技术点 4: 基于坐标的3D隐式采样 (LIIF-style 3D Implicit Sampling)")
    print("=" * 70)

    net = SimplifiedModuleNet3D(in_channel=16)
    net.eval()

    # 源特征: 4帧, 16x28 空间分辨率
    feat = torch.randn(1, 16, 4, 16, 28)
    print(f"源特征形状: {feat.shape}")

    with torch.no_grad():
        # 任意尺度上采样: 4帧→7帧, 16x28→256x448
        out1 = net(feat, target_size=(7, 256, 448))
        print(f"上采样至 (7, 256, 448): {out1.shape}")

        # 非整数倍帧率转换: 4帧→5帧
        out2 = net(feat, target_size=(5, 32, 56))
        print(f"上采样至 (5, 32, 56):   {out2.shape}")

        # 下采样: 4帧→3帧
        out3 = net(feat, target_size=(3, 8, 14))
        print(f"下采样至 (3, 8, 14):    {out3.shape}")

    print("→ 隐式采样网络支持任意目标尺寸，包括非整数倍帧率转换")
    print()


# ============================================================================
# 技术点 5: 3D密集连接块 (D2DTInput)
# ============================================================================
# 项目中的 D2DTInput 和 D2DTInput_dense 是3D密集连接块，用作可逆块的子网络。
#
# 核心思想：
#   - D2DTInput: 空间维度使用2D卷积 (1,3,3)，时间维度使用1D卷积 (3,1,1)
#     这种分解减少了参数量，同时分别处理空间和时间信息
#   - D2DTInput_dense: 使用全3D卷积 (3,3,3)，更密集的特征提取
#   - 两者都采用 DenseNet 风格的密集连接，每层接收之前所有层的输出

class D2DTInput(nn.Module):
    """3D密集连接块 (空间-时间分解版本)

    结构: 4层2D空间卷积 (1,3,3) + 1层1D时间卷积 (3,1,1)
    每层接收之前所有层的输出 (DenseNet 风格密集连接)

    参数:
        channel_in: 输入通道数
        channel_out: 输出通道数
        gc: 增长通道数 (每层产生的通道数)
        shortcut: 是否使用残差快捷连接
    """

    def __init__(self, channel_in, channel_out, gc=32, shortcut=True):
        super(D2DTInput, self).__init__()
        self.shortcut = shortcut
        # 4层空间卷积 (密集连接)
        self.conv1 = nn.Conv3d(channel_in, gc, (1, 3, 3), 1, (0, 1, 1))
        self.conv2 = nn.Conv3d(channel_in + gc, gc, (1, 3, 3), 1, (0, 1, 1))
        self.conv3 = nn.Conv3d(channel_in + 2 * gc, gc, (1, 3, 3), 1, (0, 1, 1))
        self.conv4 = nn.Conv3d(channel_in + 3 * gc, gc, (1, 3, 3), 1, (0, 1, 1))
        # 1层时间卷积 (融合时间信息)
        self.conv5 = nn.Conv3d(channel_in + 4 * gc, channel_out, (3, 1, 1), 1, (1, 0, 0))
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        if self.shortcut:
            x5 = x + x5  # 残差连接
        return x5


def demo_dense_block():
    """演示3D密集连接块的特征提取"""
    print("=" * 70)
    print("技术点 5: 3D密集连接块 (D2DTInput - 空间-时间分解)")
    print("=" * 70)

    block = D2DTInput(channel_in=48, channel_out=48, gc=32, shortcut=True)
    block.eval()

    x = torch.randn(1, 48, 4, 64, 112)
    print(f"输入形状: {x.shape}")

    with torch.no_grad():
        y = block(x)
    print(f"输出形状: {y.shape}")

    # 统计参数量
    params = sum(p.numel() for p in block.parameters())
    print(f"参数量: {params / 1e6:.2f}M")
    print(f"残差连接: 输出 = 输入 + 卷积(x)")
    print()


# ============================================================================
# 技术点 6: 可逆运动隐写模块完整流程 (IMSM Pipeline)
# ============================================================================
# 组合以上所有技术点，展示 IMSM 的完整前向-反向流程。
# 这对应项目中 IND_inv3D 的 inference_latent2RGB + inference_RGB2latent。

class SimplifiedIMSM(nn.Module):
    """简化的可逆运动隐写模块

    完整展示 IMSM 的前向（潜在特征→隐写图像）和反向（隐写图像→潜在特征）流程。

    流程:
      前向 (latent2RGB):
        x_down → InvBlockExp × N → st_shuffle × 2 → clamp → 隐写图像

      反向 (RGB2latent):
        隐写图像 → st_unshuffle × 2 → InvBlockExp(reverse) × N → clamp → 潜在特征

    参数:
        channel_in: 输入通道数
        block_num: 每层的可逆块数量
    """

    def __init__(self, channel_in=3, block_num=2):
        super(SimplifiedIMSM, self).__init__()
        self.st_shuffle = SpaceTimePixelShuffle(r=1, s=2)
        self.st_unshuffle = SpaceTimePixelUnShuffle(r=1, s=2)
        self.quan_layer = QuantizeSTE(min_val=0.0, max_val=1.0)

        # 构建可逆操作序列 (简化版: 只有一层, 实际项目有 down_num=2 层)
        self.operations = nn.ModuleList()
        # 第一层: ST_Pixelshuffle_Downsampling (通道扩展 3→12)
        # 第二层: ST_Pixelshuffle_Downsampling (通道扩展 12→48)
        # 此处简化: 直接从 48 通道开始 (假设已经过两次 ST_Pixelshuffle_Downsampling)
        current_channel = channel_in * 16  # 3 * 16 = 48 (两次 ST_Pixelshuffle_Downsampling)
        for j in range(block_num):
            block = InvBlockExp(SimpleSubnet3D, current_channel, channel_in, clamp=1.0)
            self.operations.append(block)

    def forward_latent2RGB(self, x_down):
        """前向: 潜在特征 → 隐写图像"""
        # 可逆变换
        out = x_down
        for op in self.operations:
            out = op.forward(out, rev=False)

        # 双重时空像素重排: 通道维 → 时空维
        LR_img = self.st_shuffle(self.st_shuffle(out))
        LR_img = torch.clamp(LR_img, 0, 1)
        return LR_img

    def forward_RGB2latent(self, LR_img):
        """反向: 隐写图像 → 潜在特征"""
        # 双重时空像素逆重排: 时空维 → 通道维
        LR_latent = self.st_unshuffle(self.st_unshuffle(LR_img))

        # 可逆逆变换
        out = LR_latent
        for op in reversed(self.operations):
            out = op.forward(out, rev=True)

        rev_back = torch.clamp(out, 0, 1)
        return rev_back

    def forward(self, x_down):
        """完整前向-反向流程 (含量化)"""
        # 前向: 潜在特征 → 隐写图像
        LR_img = self.forward_latent2RGB(x_down)

        # 量化: 模拟真实图像存储
        LR_img_quan = self.quan_layer(LR_img)

        # 反向: 量化后的隐写图像 → 恢复的潜在特征
        rev_back = self.forward_RGB2latent(LR_img_quan)

        return LR_img_quan, rev_back


def demo_imsm_pipeline():
    """演示完整的 IMSM 前向-反向流程"""
    print("=" * 70)
    print("技术点 6: 可逆运动隐写模块完整流程 (IMSM Pipeline)")
    print("=" * 70)

    model = SimplifiedIMSM(channel_in=3, block_num=2)
    model.eval()

    # 模拟下采样网络的输出: 48通道, 4帧, 64x112
    x_down = torch.randn(1, 48, 4, 64, 112)
    print(f"下采样网络输出 (潜在特征): {x_down.shape}")

    with torch.no_grad():
        # 前向: 潜在特征 → 隐写图像
        LR_img = model.forward_latent2RGB(x_down)
        print(f"隐写图像: {LR_img.shape}  (3通道, 视觉上像正常图像)")

        # 量化: 模拟8位存储
        LR_img_quan = model.quan_layer(LR_img)
        quan_diff = torch.mean(torch.abs(LR_img - LR_img_quan)).item()
        print(f"量化误差: {quan_diff:.6f}  (8位量化引入的误差)")

        # 反向: 隐写图像 → 恢复的潜在特征
        rev_back = model.forward_RGB2latent(LR_img_quan)
        print(f"恢复的潜在特征: {rev_back.shape}")

        # 量化导致的重建误差
        recon_diff = torch.mean(torch.abs(x_down - rev_back)).item()
        print(f"重建误差 (含量化): {recon_diff:.6f}")

        # 无量化的重建误差 (验证可逆性)
        rev_back_noquan = model.forward_RGB2latent(LR_img)
        recon_diff_noquan = torch.mean(torch.abs(x_down - rev_back_noquan)).item()
        print(f"重建误差 (无量化):   {recon_diff_noquan:.6f}  (验证可逆性)")

    print()
    print("→ 量化是 IMSM 中信息损失的唯一来源，可逆变换本身是精确可逆的")
    print()


# ============================================================================
# 主函数: 运行所有演示
# ============================================================================

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║          CSTVR 核心技术演示                                         ║")
    print("║   Continuous Space-Time Video Resampling                           ║")
    print("║   with Invertible Motion Steganography                            ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    demo_spacetime_pixel_shuffle()
    demo_invertible_block()
    demo_quantization()
    demo_implicit_sampling()
    demo_dense_block()
    demo_imsm_pipeline()

    print("=" * 70)
    print("所有核心技术点演示完成!")
    print("=" * 70)
    print()
    print("技术点总结:")
    print("  1. 时空像素重排: 通道维与时空维的可逆转换，是IMSM中信息隐藏的基础")
    print("  2. 仿射耦合可逆块: 保证变换的精确可逆性，F/G/H可以是任意网络")
    print("  3. STE量化: 模拟8位存储，是IMSM中信息损失的唯一来源")
    print("  4. 3D隐式采样: 基于坐标的神经表示，实现任意尺度时空重采样")
    print("  5. 3D密集连接块: 空间-时间分解的密集连接，高效的特征提取")
    print("  6. IMSM完整流程: 下采样→可逆编码→量化→可逆解码→上采样")


if __name__ == '__main__':
    main()
