# CSTVR 项目代码阅读指南

本项目是一个基于 3D 可逆神经网络（INN）和 3D 卷积的视频超分辨率/重缩放（Video Rescaling）项目。为了更好地理解代码逻辑，建议按照以下顺序阅读文件：

## 1. 基础工具与配置 (Utils)
首先了解项目的架构基础，包括模块注册机制和配置解析。
- [registry.py](file:///c:/codes/python/items_1/CSTVR/src/utils/registry.py): 实现了模块注册表，用于解耦模型组件。
- [options.py](file:///c:/codes/python/items_1/CSTVR/src/utils/options.py): 负责解析 YAML 配置文件。
- [model_utils.py](file:///c:/codes/python/items_1/CSTVR/src/utils/model_utils.py): 包含模型权重初始化和参数计算等通用函数。

## 2. 核心模块组件 (Modules)
了解构建网络的基本单元，特别是 3D 卷积和可逆块。
- [general_module.py](file:///c:/codes/python/items_1/CSTVR/src/module/general_module.py): 包含 3D 残差块、空时像素重组（SpaceTimePixelShuffle）等基础层。
- [inv_module.py](file:///c:/codes/python/items_1/CSTVR/src/module/inv_module.py): 核心可逆模块实现，包括 Haar 变换和仿射耦合层（InvBlockExp）。
- [Quantization.py](file:///c:/codes/python/items_1/CSTVR/src/module/Quantization.py): 实现了直通估计器（STE）量化，用于模拟 8 位图像存储。
- [sample_module.py](file:///c:/codes/python/items_1/CSTVR/src/module/sample_module.py): 基于坐标的 3D 隐式采样模块，用于实现任意尺度的缩放。
- [my_3D_module.py](file:///c:/codes/python/items_1/CSTVR/src/module/my_3D_module.py): 包含 3D Patch Embedding 和 3D 窗口自注意力机制。

## 3. 网络架构 (Architectures)
查看如何将上述组件组合成完整的重缩放网络。
- [InvNN_3D_arch.py](file:///c:/codes/python/items_1/CSTVR/src/arch/InvNN_3D_arch.py): 组合多个可逆块形成的 3D 可逆神经网络。
- [Mynet_arch.py](file:///c:/codes/python/items_1/CSTVR/src/arch/Mynet_arch.py): 定义了标准的 RescalerNet，包含下采样（STREV_down）和上采样（STREV_up）分支。
- [Mynet_mix_arch.py](file:///c:/codes/python/items_1/CSTVR/src/arch/Mynet_mix_arch.py): 混合架构版本，结合了卷积和 Transformer 块。
- [IMSM.py](file:///c:/codes/python/items_1/CSTVR/src/arch/IMSM.py): 完整的可逆运动隐写模块（Invertible Motion Steganography Module）流程。

## 4. 数据加载 (Data)
了解数据是如何输入到模型中的。
- [core_bicubic.py](file:///c:/codes/python/items_1/CSTVR/src/data/core_bicubic.py): 高效的 PyTorch 双三次插值实现。
- [vimeo_seq_dataset.py](file:///c:/codes/python/items_1/CSTVR/src/data/vimeo_seq_dataset.py): 加载 Vimeo-90K 数据集的 7 帧序列。
- [vid4_dataset.py](file:///c:/codes/python/items_1/CSTVR/src/data/vid4_dataset.py): 加载常用的 Vid4 测试集。
- [SPMCS.py](file:///c:/codes/python/items_1/CSTVR/src/data/SPMCS.py): 支持任意尺度缩放的数据集加载。

## 5. 测试与演示 (Test & Eval)
最后通过演示脚本和测试脚本了解模型的实际运行和评估方法。
- [cstvr_core_demo.py](file:///c:/codes/python/items_1/CSTVR/src/test/cstvr_core_demo.py): **强烈推荐首先阅读**，这是一个独立运行的脚本，通过简化代码展示了项目的所有核心技术点。
- [demo.py](file:///c:/codes/python/items_1/CSTVR/src/test/demo.py): 使用预训练模型对 `test_input` 中的图像进行完整推理的演示。
- [vimeo7_test.py](file:///c:/codes/python/items_1/CSTVR/src/test/vimeo7_test.py): 在 Vimeo-90K 上进行性能测试。
- [psnr_ssim.py](file:///c:/codes/python/items_1/CSTVR/src/metrics/psnr_ssim.py): 指标计算函数。
- [eval.py](file:///c:/codes/python/items_1/CSTVR/src/eval/eval.py): 批量评估脚本。

---
希望这份指南能帮助你快速上手 CSTVR 项目！
