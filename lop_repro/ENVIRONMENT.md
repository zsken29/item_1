# 运行环境说明

更新时间：2026-04-28

本项目默认使用 Conda 环境 `cv_env`。

Python 解释器：

```text
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe
```

## 当前关键版本

| 组件 | 版本 |
| --- | --- |
| Python | 3.10.20 |
| PyTorch | 2.7.0+cu128 |
| torchvision | 0.22.0+cu128 |
| torchaudio | 2.7.0+cu128 |
| FlashAttention2 | 2.7.4.post1 |
| transformers | 5.6.0.dev0 |
| safetensors | 0.7.0 |
| bitsandbytes | 0.49.2 |
| qwen-vl-utils | 0.0.14 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Driver CUDA | 12.8 |

## 已处理的问题

- 清理了 `cv_env` 中残留的 `~uggingface_hub*` 坏包目录。
- 将 `setuptools` 调整为 `81.0.0`，满足 PyTorch 依赖约束。
- 将 PyTorch 调整为 `2.7.0+cu128`，匹配当前 CUDA/FlashAttention2 Windows wheel。
- 安装并验证了 FlashAttention2 CUDA kernel。
- 验证了 bitsandbytes CUDA 线性层。
- 验证了 Qwen2.5-VL 配置和 processor 加载。
- 验证了 InternVL2_5-1B CUDA 加载、单样本推理和 FFN activation hook。

## 验证命令

```powershell
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.check_environment --strict
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m unittest discover -v
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.inspect_data
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.inspect_model models/Qwen2.5-VL-7B-Instruct
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.inspect_model models/InternVL2_5-1B
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.smoke_internvl --layers 2 --max-new-tokens 8
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.smoke_prune_internvl --layers 2 --ratio 0.1 --max-new-tokens 4
C:\Users\ZSQ\anaconda3\envs\cv_env\python.exe -m scripts.evaluate_dataset --dataset mme --runtime internvl --model-dir models/InternVL2_5-1B --limit 1 --max-new-tokens 4 --output outputs/reports/internvl_dataset/mme_limit1.json
```

## Hugging Face 缓存约定

加载 Hugging Face remote code 模型时，缓存必须放在工作区内，避免写入用户主目录：

```powershell
$env:HF_HOME = Join-Path (Get-Location) 'outputs\hf_home'
$env:HF_MODULES_CACHE = Join-Path $env:HF_HOME 'modules'
$env:TRANSFORMERS_CACHE = Join-Path $env:HF_HOME 'transformers'
```

`lop/models/internvl.py` 已在加载 InternVL 前设置上述缓存路径。
