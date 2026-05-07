from __future__ import annotations

from dataclasses import dataclass

import torch

# 记录一次 activation 采集的汇总信息：覆盖了多少层、采集了多少 token、每层神经元维度
@dataclass(frozen=True)
class ActivationSummary:
    layer_count: int
    token_count: int
    dimensions: tuple[int, ...]


# 对多模态大语言模型（MLLM）的 FFN 层进行 activation 采集，用于计算神经元重要性。
# 论文 Appendix A.2 使用 FFN down_proj 的输入 activation 衡量神经元重要性：
#   - 位置：每层 mlp.down_proj 的输入，最后一维恰好对应 FFN 中间层神经元（intermediate neurons）
#   - 统计量：RMS（论文主流程）或 variance（FLAP 基线所需）
class FfnActivationCollector:
    def __init__(self, model: torch.nn.Module, module_paths: list[str]) -> None:
        self._model = model
        self._module_paths = module_paths        # 每层 hook 的模块路径列表
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._sums: dict[str, torch.Tensor] = {}          # 按路径存储激活值的逐神经元累加和
        self._sum_squares: dict[str, torch.Tensor] = {}   # 按路径存储激活值的逐神经元平方和
        self._counts: dict[str, int] = {}                  # 按路径存储累计 token 数

    # 进入上下文管理器时，为每个模块路径注册 forward pre-hook，开始采集
    def __enter__(self) -> "FfnActivationCollector":
        for path in self._module_paths:
            module = self._model.get_submodule(path)
            self._handles.append(module.register_forward_pre_hook(self._make_hook(path)))
        return self

    # 退出上下文管理器时，移除所有 hook，停止采集
    def __exit__(self, exc_type, exc, traceback) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    # 计算每层神经元的 RMS 重要性（论文默认指标，Eq.24-25）：
    #   importance[j] = sqrt( sum(x_j^2) / N )
    # 其中 j 为神经元索引，N 为该层累计 token 数
    def importance(self) -> dict[str, torch.Tensor]:
        result = {}
        for path in self._module_paths:
            if path not in self._sum_squares:
                raise RuntimeError(f"no activation captured for {path}")
            result[path] = torch.sqrt(self._sum_squares[path] / self._counts[path])
        return result

    # 计算每层神经元的 activation variance（用于 FLAP WIFV 基线）：
    #   variance[j] = E[x_j^2] - E[x_j]^2
    # 结果做 clamp(min=0) 避免浮点误差导致负值
    def variance(self) -> dict[str, torch.Tensor]:
        result = {}
        for path in self._module_paths:
            if path not in self._sum_squares:
                raise RuntimeError(f"no activation captured for {path}")
            count = self._counts[path]
            mean = self._sums[path] / count
            second_moment = self._sum_squares[path] / count
            result[path] = torch.clamp(second_moment - mean * mean, min=0.0)
        return result

    # 返回本次采集的汇总统计：覆盖层数、各层维度、总 token 数
    def summary(self) -> ActivationSummary:
        dims = tuple(int(self._sum_squares[path].numel()) for path in self._module_paths)
        token_count = sum(self._counts.values())
        return ActivationSummary(layer_count=len(self._module_paths), token_count=token_count, dimensions=dims)

    # 创建一个 forward pre-hook，在模块前向传播前被触发。
    # 输入 tensor 的第一维是序列长度，最后一维是 intermediate_size（神经元维度），
    # 按最后一维累加一阶矩（sums）和二阶矩（sum_squares），跨批累积。
    def _make_hook(self, path: str):
        def hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if not inputs:
                raise RuntimeError(f"{path} received no input")
            activation = inputs[0].detach().float()
            if activation.ndim < 2:
                raise RuntimeError(f"{path} activation must have at least 2 dimensions")
            # 将除最后一维外的维度展平，得到 (total_tokens, intermediate_size)
            flat = activation.reshape(-1, activation.shape[-1])
            sums = flat.sum(dim=0).cpu()
            values = (flat * flat).sum(dim=0).cpu()
            if path in self._sum_squares:
                self._sums[path] += sums
                self._sum_squares[path] += values
                self._counts[path] += flat.shape[0]
            else:
                self._sums[path] = sums
                self._sum_squares[path] = values
                self._counts[path] = flat.shape[0]

        return hook
