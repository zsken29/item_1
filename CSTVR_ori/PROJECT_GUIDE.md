# CSTVR 项目

> **论文**: "Continuous Space-Time Video Resampling with Invertible Motion Steganography"

***

## 一、项目简介

### 1.1 研究背景

时空视频重采样（Space-Time Video Resampling）旨在同时进行时空下采样和上采样过程，以实现高质量的视频重建。该领域存在以下主要挑战：

- **运动信息保留**：如何在时间重采样过程中保留运动信息，同时避免模糊伪影
- **灵活的重采样因子**：如何实现灵活的时间和空间重采样因子（如非整数倍帧率转换）

### 1.2 核心创新

本项目提出以下创新解决方案：

| 创新点                  | 描述                                                                                      |
| -------------------- | --------------------------------------------------------------------------------------- |
| **可逆运动隐写模块 (IMSM)** | 将高帧率视频的潜在特征（含运动信息）以视觉不可见的方式嵌入到低帧率下采样帧中，其可逆性允许恢复潜在特征以重建高帧率视频 |
| **3D隐式特征调制**         | 基于坐标的3D隐式神经表示（LIIF扩展），实现连续时空重采样，支持非整数倍帧率转换（如 30 FPS ↔ 24 FPS） |
| **灵活的训练策略**          | 支持多种帧率转换场景，包括固定尺度（时间2x/空间1x）和连续尺度重采样                                  |

### 1.3 项目结构概览

```
CSTVR/
├── src/
│   ├── arch/          # 网络架构定义
│   ├── module/        # 核心功能模块
│   ├── data/          # 数据集加载器
│   ├── test/          # 测试脚本
│   ├── eval/          # 评估工具
│   ├── metrics/       # 评价指标
│   └── utils/         # 工具函数
├── test_input/        # 测试输入数据（Vimeo-90K样本，每个序列7帧）
├── pic/               # 文档图片
└── archived/          # 预训练权重（需下载）
```

***

## 二、核心模块详解

### 2.1 网络架构层 (`src/arch/`)

| 文件                  | 类名                | 功能描述                                                                                                          |
| ------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------- |
| `IMSM.py`           | `IND_inv3D`       | **可逆运动隐写模块**的核心实现。包含下采样网络（`STREV_down`）、3D可逆神经网络（`my_InvNN`）、量化层（`Quantize_ste`）及双重时空像素重排，实现潜在特征的嵌入与恢复 |
| `Mynet_arch.py`     | `STREV_down`      | **时空下采样网络**。3D卷积特征提取 → 3D残差块 → 隐式采样模块（`ModuleNet3D`），实现任意尺度下采样                                              |
| `Mynet_arch.py`     | `STREV_up`        | **时空上采样网络**（基础版）。3D卷积 → 时空像素逆重排 → 3D密集块 → 通道融合 → 时空像素重排 → 隐式采样模块，将潜在特征上采样重建为高帧率视频                          |
| `Mynet_arch.py`     | `RescalerNet`     | **整体重缩放网络**（基础版）。组合 `STREV_down` + `Quantize_ste` + `STREV_up`，用于固定尺度（时间2x/空间1x）重采样                         |
| `Mynet_mix_arch.py` | `STREV_down`      | **时空下采样网络**（与 `Mynet_arch.py` 中相同）                                                                          |
| `Mynet_mix_arch.py` | `STREV_up`        | **时空上采样网络**（增强版）。在基础版上集成 Video Swin Transformer（`BasicLayer`），支持可配置的像素重排和残差快捷连接                             |
| `Mynet_mix_arch.py` | `Rescaler_MixNet` | **混合重缩放网络**。组合 `STREV_down` + `Quantize_ste` + 增强版 `STREV_up`，支持更灵活的时空重采样因子，含 `control_rate` 控制参数           |
| `InvNN_3D_arch.py`  | `my_InvNN`        | **3D可逆神经网络**。由 `ST_Pixelshuffle_Downsampling` 和 `InvBlockExp` 交替堆叠构成，实现可逆的前向/反向传播，用于隐写信息的编码与解码        |

> **注意**：`Mynet_mix_arch.py` 中的 `STREV_up` 使用 `basicsr.utils.registry.ARCH_REGISTRY` 注册，而其他架构文件使用项目自身的 `utils.registry.ARCH_REGISTRY`，存在注册器不一致。

### 2.2 功能模块层 (`src/module/`)

#### 2.2.1 可逆变换模块 (`inv_module.py`)

| 组件                            | 功能描述                                                                 |
| ----------------------------- | -------------------------------------------------------------------- |
| `HaarDownsampling`            | 2D哈尔小波下采样，可逆变换，将空间分辨率减半、通道数扩4倍                                      |
| `Pixelshuffle_Downsampling`   | 2D像素重排下采样（PixelUnshuffle），可逆变换                                      |
| `ST_Pixelshuffle_Downsampling`| **3D时空像素重排下采样**（r=1, s=2），可逆变换，用于 `my_InvNN` 中的通道扩展                 |
| `InvBlockExp`                 | **可逆块扩展**，核心可逆计算单元。采用仿射耦合层（Affine Coupling），包含 F/G/H 三个子网络，支持 `spilt_mode` 通道翻转 |
| `D2DTInput`                   | 3D密集连接块（2D空间卷积 + 1D时间卷积），带可选残差快捷连接，用作 `InvBlockExp` 的子网络构造器（`Mynet_mix_arch` 版本） |
| `D2DTInput_dense`             | 3D密集连接块（全3D卷积），用作 `IND_inv3D` 中 `my_InvNN` 的子网络构造器                   |
| `DenseBlock`                  | 2D密集连接块，5层密集卷积                                                      |
| `HaarTransform`               | 哈尔小波变换（`register_buffer` 实现），与前向传播版本功能相同但使用更优的参数管理方式                |
| `Haartrans` / `Haartrans_back`| 函数式哈尔小波正/逆变换工具                                                      |

#### 2.2.2 通用模块 (`general_module.py`)

| 组件                                | 功能描述                                              |
| --------------------------------- | ------------------------------------------------- |
| `ResidualBlock3D_NoBN`            | 3D残差块（无批归一化），使用 PReLU 激活                          |
| `DepthwiseSeparableConv3d`        | 深度可分离3D卷积，减少参数量                                   |
| `DepthwiseTransSeparableConv3d`   | 深度可分离3D转置卷积，用于上采样                                 |
| `ResidualBlock3D_depthwise_NoBN`  | 深度可分离3D残差块                                        |
| `ResBlock`                        | 2D残差块，含侧通道（side_channels）处理                      |
| `Encoder`                         | 2D金字塔编码器，3级下采样                                    |
| `ResidualBlockNoBN`               | 2D残差块（无批归一化）                                      |
| `make_layer`                      | 通用层堆叠工具函数，将同一模块重复堆叠为 `nn.Sequential`             |
| `ResidualBlocksWithInputConv`     | 带输入卷积的2D残差块堆叠                                     |
| `SpaceTimePixelShuffle`           | **时空像素重排**（上采样），参数 r（时间因子）和 s（空间因子），将通道维转换为时空维 |
| `SpaceTimePixelUnShuffle`         | **时空像素逆重排**（下采样），将时空维转换为通道维                       |

#### 2.2.3 隐式采样模块 (`sample_module.py`)

| 组件              | 功能描述                                                                                   |
| --------------- | -------------------------------------------------------------------------------------- |
| `make_coord`    | 2D坐标生成函数，在 [-1, 1] 范围内生成网格中心坐标                                                       |
| `make_coord_3d` | 3D坐标生成函数，支持 `row_first`（ij）和 `xy` 两种索引模式                                             |
| `ModuleNet2d`   | **2D隐式采样网络**。基于 LIIF（Local Implicit Image Function），通过最近邻特征采样 + 相对坐标编码 + cell 编码实现任意尺度2D重采样 |
| `ModuleNet3D`   | **3D隐式采样网络**。LIIF 的3D扩展版本，在3D体素空间中进行8邻域特征采样 + 3D相对坐标编码 + cell 编码，实现任意尺度3D时空重采样        |

> **核心机制**：`ModuleNet3D` 通过 `F.grid_sample` 获取最近邻特征，计算查询点与最近邻格点的相对坐标，结合 cell（感受野）信息，经深度可分离3D卷积 + MLP 预测目标位置的像素值，并叠加双线性插值的快捷连接。

#### 2.2.4 量化模块 (`Quantization.py`)

| 组件                        | 功能描述                                                   |
| ------------------------- | ------------------------------------------------------ |
| `Quant`                   | 直通估计器（STE）量化，`torch.autograd.Function` 实现，前向量化为8位、反向直通 |
| `Quantization`            | `Quant` 的 `nn.Module` 封装                               |
| `Quantize_ste`            | **项目主要使用的量化层**。STE量化，支持自定义 min/max 范围，使用 `detach()` 技巧实现直通梯度 |
| `DifferentiableClipping`  | 可微分裁剪，自定义前向/反向传播，边界处梯度放大为2                            |
| `DifferentiableQuantization` | `DifferentiableClipping` 的 `nn.Module` 封装              |

#### 2.2.5 Video Swin Transformer 3D (`my_3D_module.py`)

| 组件                          | 功能描述                                                       |
| --------------------------- | ---------------------------------------------------------- |
| `PatchEmbed3D`              | 3D视频块嵌入，将视频分割为3D patch 并线性投影                              |
| `WindowAttention3D`         | 3D窗口多头自注意力，含相对位置偏置                                       |
| `SwinTransformerBlock3D`    | 3D Swin Transformer 块，支持移位窗口（SW-MSA）                      |
| `BasicLayer`                | Swin Transformer 基础层，堆叠多个 `SwinTransformerBlock3D`，含3D卷积残差连接 |
| `PatchMerging`              | Patch 合并层，2x空间下采样 + 通道扩展                                  |
| `SwinTransformer3D`         | 完整的3D Swin Transformer 骨干网络                               |
| `window_partition/reverse`  | 3D窗口分割/合并工具函数                                             |
| `compute_mask`              | 移位窗口注意力掩码计算（带缓存）                                          |

> **依赖**：`my_3D_module.py` 依赖 `timm`（`DropPath`, `trunc_normal_`）和 `einops`（`rearrange`）库。

### 2.3 数据处理层 (`src/data/`)

| 文件                     | 类/数据集            | 描述                                                                                          |
| ---------------------- | ---------------- | ------------------------------------------------------------------------------------------- |
| `vid4_dataset.py`      | `Vid4`           | 4个场景（calendar, city, foliage, walk），用于固定尺度测试。使用项目自身 `DATASET_REGISTRY` 注册                |
| `vimeo_seq_dataset.py` | `Vimeo_SepTuplet`| Vimeo-90K 大规模视频数据集（7帧序列），用于训练和测试。支持数据增强（翻转/旋转）和随机裁剪。使用 `basicsr` 的 `DATASET_REGISTRY` 注册 |
| `SPMCS.py`             | `SPMCS_arb`      | SPMCS 数据集，用于连续时空重采样测试，支持任意空间/时间缩放因子                                                        |
| `core_bicubic.py`      | -                | 双三次插值工具（与 MATLAB `imresize` 结果一致），提供 `imresize` 函数支持任意尺度缩放                                |

> **注意**：`Vimeo_SepTuplet` 使用 `basicsr.utils.registry.DATASET_REGISTRY` 注册，与项目自身的注册器不一致。

### 2.4 测试与评估 (`src/test/`, `src/eval/`, `src/metrics/`)

| 文件                          | 功能                                                                                      |
| --------------------------- | --------------------------------------------------------------------------------------- |
| `test/demo.py`              | 快速演示脚本，执行时间2x/空间1x重采样。加载 `IND_inv3D` + `RescalerNet`，对 `test_input/` 中的样本进行推理并保存结果 |
| `test/vid4_test.py`         | Vid4数据集固定尺度测试，支持分块推理（y_grid=1, x_grid=2）以适应大分辨率，计算 PSNR/SSIM（RGB & Y通道）             |
| `test/vimeo7_test.py`       | Vimeo-90K数据集测试，支持可配置的 time_factor 和 scale_factor，同时评估下采样和重建质量                      |
| `test/SPMCS_contin_test.py` | 连续重采样测试（非整数倍帧率转换），使用 `basicsr.metrics.psnr_ssim` 计算指标，支持多尺度列表测试                     |
| `eval/eval.py`              | 批量评估脚本，包含 `test_vid()`、`com_vimeo()`、`com_SPMCS()` 三个评估函数，计算 PSNR/SSIM             |
| `metrics/psnr_ssim.py`      | PSNR和SSIM指标计算实现，依赖 `basicsr.metrics.metric_util`，支持 Y 通道测试                          |

### 2.5 工具函数 (`src/utils/`)

| 文件               | 功能                                                                                      |
| ---------------- | --------------------------------------------------------------------------------------- |
| `options.py`     | YAML配置文件加载与解析（`yaml_load`），训练选项解析（`parse_options`），支持分布式训练配置、随机种子设置、强制YAML更新等 |
| `registry.py`    | 模块注册器模式，定义5个全局注册器：`DATASET_REGISTRY`、`ARCH_REGISTRY`、`MODEL_REGISTRY`、`LOSS_REGISTRY`、`METRIC_REGISTRY` |
| `model_utils.py` | 模型参数统计（`get_model_total_params`，单位M）、Kaiming/Xavier权重初始化                              |

***

## 三、技术流程

### 3.1 整体流程图（IMSM 路径）

以下为使用 `IND_inv3D`（IMSM模块）的完整推理流程，对应 `demo.py` 中的调用方式：

```
┌─────────────────────────────────────────────────────────────────┐
│                     输入视频 (B, 3, 7, H, W)                    │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  [STREV_down] 时空下采样网络 (RescalerNet.inference_down)        │
│  1. 3D卷积特征提取 (first_block)                                │
│  2. 3D残差块特征提取 (feat_extractor)                           │
│  3. ModuleNet3D 隐式采样至目标尺寸                               │
│  → 输出: x_down (B, 3, 4, H/2, W)                             │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  [IND_inv3D.inference_latent2RGB] 可逆隐写编码                    │
│  1. my_InvNN 前向: ST_Pixelshuffle_Downsampling × 2层           │
│     + InvBlockExp × block_num × 2层 → LR_img_stack             │
│  2. SpaceTimePixelShuffle(r=1,s=2) × 2次 → 通道维转时空维        │
│  3. clamp [0,1]                                                │
│  → 输出: LR_img (B, 3, 4, H, W) 隐写图像                       │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  [uint8 量化] 模拟真实图像存储                                    │
│  - 转为 uint8 再转回 float32 (等价于8位量化)                     │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  [IND_inv3D.inference_RGB2latent] 可逆隐写解码                    │
│  1. SpaceTimePixelUnShuffle(r=1,s=2) × 2次 → 时空维转通道维     │
│  2. my_InvNN 反向: InvBlockExp(reverse) × block_num × 2层       │
│     + ST_Pixelshuffle_Downsampling(reverse) × 2层               │
│  3. clamp [0,1]                                                │
│  → 输出: rev_back (B, 3, 4, H/2, W) 恢复的潜在特征              │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  [STREV_up] 时空上采样网络 (RescalerNet.inference_up)            │
│  1. 3D卷积 (first_block)                                       │
│  2. SpaceTimePixelUnShuffle → 时空维转通道维                     │
│  3. 3D密集块特征提取 (dense3d_backbone) + 残差快捷连接            │
│  4. 通道融合 (fuse_channel_out + fuse_channel_in)               │
│  5. SpaceTimePixelShuffle → 通道维转时空维                       │
│  6. ModuleNet3D 隐式采样至目标尺寸                               │
│  → 输出: 重建视频 (B, 3, 7, H, W)                              │
└─────────────────────────────────────────────────────────────────┘
```

> **非IMSM路径**：`RescalerNet` 的 `forward` 方法直接执行 STREV_down → Quantize_ste → STREV_up，不经过 IMSM 的可逆隐写编码/解码过程。

### 3.2 关键技术点

#### 可逆运动隐写 (IMSM)

IMSM 的核心思想是：通过可逆神经网络将下采样后的潜在特征（3通道、低帧率）变换为看起来像自然图像的隐写图像，其可逆性保证了从隐写图像可以无损恢复潜在特征。

```python
# === 前向过程：潜在特征 → 隐写图像 (inference_latent2RGB) ===
x_down = self.downsample(imgs, down_size)           # STREV_down: 时空下采样
LR_img_stack, jac = self.inv_block(x_down)          # my_InvNN: 可逆变换，扩展通道
LR_img = self.st_shuffle(self.st_shuffle(LR_img_stack))  # 双重时空像素重排: 通道维→时空维
LR_img = torch.clamp(LR_img, 0, 1)                 # 裁剪至合法范围

# === 量化：模拟真实存储 ===
LR_img_uint8 = (LR_img * 255).astype(np.uint8)     # 8位量化
LR_img_quan = LR_img_uint8.astype(np.float32) / 255.0

# === 反向过程：隐写图像 → 潜在特征 (inference_RGB2latent) ===
LR_latent = self.st_unshuffle(self.st_unshuffle(LR_img_quan))  # 双重逆像素重排: 时空维→通道维
rev_back = self.inv_block(LR_latent, rev=True)     # my_InvNN 逆向: 恢复原始通道数
rev_back = torch.clamp(rev_back, 0, 1)             # 裁剪至合法范围
```

**双重时空像素重排的作用**：`SpaceTimePixelShuffle(r=1, s=2)` 每次将空间维度扩大2倍、通道数减少4倍。调用两次后，空间维度扩大4倍、通道数减少16倍。对于3通道输入，经过 `my_InvNN` 中两次 `ST_Pixelshuffle_Downsampling`（通道扩4×4=16倍至48通道），再经过两次 `st_shuffle`（通道减4×4=16倍），最终恢复为3通道、空间分辨率4倍大的隐写图像。

#### 3D可逆神经网络 (my_InvNN)

`my_InvNN` 由 `down_num` 层下采样-可逆块对组成（默认2层），每层包含：

1. `ST_Pixelshuffle_Downsampling`：时空像素逆重排，空间减半、通道扩4倍
2. `block_num` 个 `InvBlockExp`：仿射耦合可逆块

```python
# my_InvNN 结构（默认 down_num=2, block_num=2）
# 层1: ST_Pixelshuffle_Downsampling → InvBlockExp → InvBlockExp
# 层2: ST_Pixelshuffle_Downsampling → InvBlockExp → InvBlockExp
# 通道变化: 3 → 12 → 48 (经过两次 ST_Pixelshuffle_Downsampling)
```

#### 连续时空重采样 (ModuleNet3D)

`ModuleNet3D` 基于 LIIF（Local Implicit Image Function）思想扩展到3D，实现任意尺度时空重采样：

```python
# 3D坐标生成
coord = make_coord_3d(target_size)  # 生成 (T, H, W) 目标坐标

# 8邻域特征采样 + 相对坐标编码
for vt in [-1, 1]:
    for vx in [-1, 1]:
        for vy in [-1, 1]:
            # 偏移坐标 → grid_sample 获取最近邻特征
            feat_ = F.grid_sample(feat, coord_.flip(-1), mode='nearest')
            # 计算相对坐标并缩放
            rel_coord = coord - old_coord
            # 计算体积权重（3D扩展的面积权重）
            area = abs(rel_coord_t * rel_coord_h * rel_coord_w)

# 加权特征 + cell编码 → 深度可分离3D卷积 + MLP → 预测值
ret = self.fc2(F.gelu(self.fc1(self.conv00_3d(grid))))
# 叠加双线性插值快捷连接
ret = ret + F.grid_sample(shortcut, coord.flip(-1), mode='bilinear')
```

***

## 四、环境配置与使用

### 4.1 环境要求

- Python 3.8.18
- torch == 2.0.0
- torchvision == 0.15.1
- numpy == 1.22.3
- basicsr（用于 `Vimeo_SepTuplet` 数据集、`SPMCS_contin_test.py` 中的指标计算）
- timm（用于 `my_3D_module.py` 中的 `DropPath`, `trunc_normal_`）
- einops（用于 `my_3D_module.py` 中的 `rearrange`）

### 4.2 预训练权重

从以下地址下载预训练模型，放置于 `CSTVR/archived` 目录：

- 百度网盘: <https://pan.baidu.com/s/16L1WyclbxvkRSIJDImIjWQ>
- 密码: `43x5`

权重目录结构：

```
archived/
└── Tx2_Sx1_vimeo/          # 时间2x/空间1x 模型
    ├── inverter/           # 可逆隐写模块 (IND_inv3D)
    │   ├── config.yml
    │   └── model.pth
    └── rescaler/           # 重缩放网络 (RescalerNet)
        ├── config.yml
        └── model.pth
```

### 4.3 快速开始

运行演示脚本：

```bash
cd src/test
python demo.py
```

输出结果位于 `CSTVR/output/demo/`，目录结构：

```
output/demo/
├── {序列名}/
│   ├── latent/     # 潜在特征可视化（下采样网络的直接输出）
│   ├── rev/        # 逆向重建结果（从隐写图像恢复的潜在特征）
│   ├── sr/         # 超分辨率输出（最终重建的7帧视频）
│   └── stegan/     # 隐写图像（可逆隐写编码后的可视化图像）
```

### 4.4 固定尺度测试

```bash
cd src/test

# Vid4数据集测试（需指定数据集路径）
python vid4_test.py --data_dir /path/to/Vid4/GT_cp/ \
    --weight_base_p /path/to/archived/

# Vimeo-90K数据集测试（支持可配置的时空因子）
python vimeo7_test.py --time_factor 2 --scale_factor 1 \
    --data_dir /path/to/vimeo/ --weight_base_p /path/to/archived/
```

### 4.5 连续重采样测试

```bash
cd src/test
python SPMCS_contin_test.py --data_dir /path/to/SPMCS/ \
    --weight_base_p /path/to/archived/Contin/
```

***

## 五、性能表现

项目在多个数据集上进行了广泛实验，显著优于现有方法：

- 支持多种视频重采样任务（固定尺度 + 连续尺度）
- 高灵活性的帧率转换（支持非整数倍帧率转换）
- 优秀的重建质量（PSNR/SSIM 指标）

详细性能数据请参考原论文。

***

## 六、已知问题与注意事项

1. **注册器不一致**：`Mynet_mix_arch.py` 和 `vimeo_seq_dataset.py` 使用 `basicsr` 的注册器，而其他文件使用项目自身的 `utils.registry`，混合使用时需注意
2. **指标计算不一致**：`SPMCS_contin_test.py` 使用 `basicsr.metrics.psnr_ssim`，而其他测试脚本使用项目自身的 `metrics.psnr_ssim`
3. **`inv_module.py` 中存在两个 `D2DTInput` 类定义**：第一个（无 `shortcut` 参数，约第191行）和第二个（有 `shortcut` 参数，约第247行），Python 会使用后定义的版本（即带 `shortcut` 的版本）
4. **`Quantization.py` 底部包含测试代码**：`DifferentiableQuantization` 示例在模块导入时会执行，可能影响使用

***

## 七、致谢

本项目基于以下开源工作构建：

- [open-mmlab](https://github.com/open-mmlab)（basicsr 框架）
- [bicubic_pytorch](https://github.com/sanghyun-son/bicubic_pytorch)（双三次插值）
- [Video Swin Transformers](https://github.com/shoaib6174/GSOC-22-Video-Swin-Transformers)（3D Swin Transformer）
- [SelfC](https://github.com/tianyuan168326/SelfC)（自条件隐式神经表示）

感谢各位作者的代码分享！
