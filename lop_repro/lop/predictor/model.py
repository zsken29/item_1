from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

# MCTS 搜索产出的一个训练样本：(全局剪枝率, 逐层剪枝率, reward)
@dataclass(frozen=True)
class PredictorSample:
    global_ratio: float
    layer_ratios: tuple[float, ...]
    reward: float

# MLP 预测器（Appendix A.1.1 消融实验方案一）。
# 输入 target_ratio (标量) → 两层隐藏层 → sigmoid 输出 L 层剪枝率。
# 结构：1 → hidden → hidden → L，激活 ReLU，输出 sigmoid。
class MlpPruningPredictor(torch.nn.Module):
    def __init__(self, layer_count: int, hidden_size: int = 128) -> None:
        super().__init__()
        _validate_layer_count(layer_count)
        self.layer_count = layer_count
        self.hidden_size = hidden_size
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, layer_count),
        )

    def forward(self, target_ratio: torch.Tensor, teacher: torch.Tensor | None = None) -> torch.Tensor:
        return torch.sigmoid(self.net(target_ratio.reshape(-1, 1)))

# Bi-LSTM 预测器（Appendix A.1.1 消融实验方案二）。
# 输入 target_ratio → 投影后重复 L 次构造序列 → Bi-LSTM → sigmoid 输出。
# hidden_size 必须为偶数（双向各一半）。
class BiLstmPruningPredictor(torch.nn.Module):
    def __init__(self, layer_count: int, hidden_size: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        _validate_layer_count(layer_count)
        if hidden_size % 2 != 0:
            raise ValueError("BiLSTM hidden_size must be even")
        self.layer_count = layer_count
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.project = torch.nn.Linear(1, hidden_size)
        self.lstm = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.output = torch.nn.Linear(hidden_size, 1)

    def forward(self, target_ratio: torch.Tensor, teacher: torch.Tensor | None = None) -> torch.Tensor:
        batch = target_ratio.shape[0]
        token = self.project(target_ratio.reshape(batch, 1))
        sequence = token.unsqueeze(1).expand(batch, self.layer_count, self.hidden_size)
        hidden, _ = self.lstm(sequence)
        return torch.sigmoid(self.output(hidden)).squeeze(-1)

# Transformer 自回归预测器（论文默认架构，Appendix A.1.2）。
#
# 输入：目标全局剪枝率 b
# 输出：逐层剪枝率 θ_1, ..., θ_L（自回归，前一层的预测输入到下一层的位置编码中）
#
# 架构：
#   - budget_mlp: 将标量 b 映射为 hidden_size 向量，作为序列的第一个 token
#   - layer_embedding: 可学习的位置编码，每层一个 embedding 向量
#   - encoder: TransformerEncoder，使用因果 mask（自回归）
#   - output: 线性层将隐藏状态投影为标量再 sigmoid
#
# 论文默认配置：2 layers, hidden 128, 4 heads, GELU, learnable position
class AutoregressivePruningPredictor(torch.nn.Module):
    def __init__(
        self,
        layer_count: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_expansion: int = 4,
    ) -> None:
        super().__init__()
        _validate_layer_count(layer_count)
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.layer_count = layer_count
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_expansion = ffn_expansion
        self.budget_mlp = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, hidden_size),
        )
        self.layer_embedding = torch.nn.Embedding(layer_count, hidden_size)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * ffn_expansion,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = torch.nn.Linear(hidden_size, 1)

    # 训练时用 teacher forcing（传入真实的逐层剪枝率作为前层输入），
    # 推理时用自回归（前层预测值 detach 后馈入下一层）
    def forward(self, target_ratio: torch.Tensor, teacher: torch.Tensor | None = None) -> torch.Tensor:
        if teacher is None:
            return self._infer(target_ratio)
        previous = torch.cat([torch.zeros_like(teacher[:, :1]), teacher[:, :-1]], dim=1)
        return self._predict_from_previous(target_ratio, previous)

    # 推理模式：逐层自回归预测。
    # 第 0 层输入全零向量（无历史），每层预测后将结果 detach 防止梯度流动。
    def _infer(self, target_ratio: torch.Tensor) -> torch.Tensor:
        batch = target_ratio.shape[0]
        previous = torch.zeros(batch, self.layer_count, device=target_ratio.device)
        outputs = []
        for layer in range(self.layer_count):
            prediction = self._predict_from_previous(target_ratio, previous)
            current = prediction[:, layer : layer + 1]
            outputs.append(current)
            previous[:, layer : layer + 1] = current.detach()
        return torch.cat(outputs, dim=1)

    # 核心前向计算：
    #   1. budget token: b → budget_mlp → (B, 1, H)
    #   2. ratio tokens: 前层剪枝率 × 对应位置embedding → (B, L, H)
    #   3. 拼接为序列 [budget, ratio_0, ..., ratio_{L-1}]
    #   4. 因果 Transformer 编码后取除 budget 外的 L 个输出
    #   5. 线性投影 + sigmoid 得到 (B, L) 剪枝率
    def _predict_from_previous(self, target_ratio: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        batch = target_ratio.shape[0]
        budget = self.budget_mlp(target_ratio.reshape(batch, 1)).unsqueeze(1)
        positions = torch.arange(self.layer_count, device=target_ratio.device)
        embeddings = self.layer_embedding(positions).unsqueeze(0)
        ratio_tokens = previous.unsqueeze(-1) * embeddings
        sequence = torch.cat([budget, ratio_tokens], dim=1)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(sequence.shape[1], device=sequence.device)
        hidden = self.encoder(sequence, mask=mask)
        return torch.sigmoid(self.output(hidden[:, 1:, :])).squeeze(-1)

# 根据架构名称构建对应的预测器
def build_predictor(
    architecture: str,
    layer_count: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    num_heads: int = 4,
) -> torch.nn.Module:
    if architecture == "transformer":
        return AutoregressivePruningPredictor(
            layer_count=layer_count,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
        )
    if architecture == "bilstm":
        return BiLstmPruningPredictor(layer_count=layer_count, hidden_size=hidden_size, num_layers=num_layers)
    if architecture == "mlp":
        return MlpPruningPredictor(layer_count=layer_count, hidden_size=hidden_size)
    raise ValueError(f"unknown predictor architecture: {architecture}")

# 从 MCTS 输出的 JSONL 文件中加载训练样本
def load_search_samples(path: str | Path) -> list[PredictorSample]:
    sample_path = Path(path)
    if not sample_path.is_file():
        raise FileNotFoundError(f"missing search samples: {sample_path}")
    samples = []
    with sample_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            item = json.loads(line)
            ratios = tuple(float(value) for value in item["layer_ratios"])
            if not ratios:
                raise ValueError(f"{sample_path}:{line_number}: empty layer_ratios")
            samples.append(
                PredictorSample(
                    global_ratio=float(item["global_ratio"]),
                    layer_ratios=ratios,
                    reward=float(item["reward"]),
                )
            )
    if not samples:
        raise ValueError(f"{sample_path}: no samples")
    layer_count = len(samples[0].layer_ratios)
    for sample in samples:
        if len(sample.layer_ratios) != layer_count:
            raise ValueError(f"{sample_path}: inconsistent layer ratio lengths")
    return samples

# 训练预测器。
#
# 损失函数：
#   - ratio_loss: 加权 MSE(预测剪枝率, MCTS 搜索的剪枝率)，权重为 reward 归一化值
#   - budget_loss: (预测均值 - 目标剪枝率)^2，确保预测满足全局预算
#   - total = ratio_loss + budget_loss
def train_predictor(
    samples: list[PredictorSample],
    hidden_size: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    architecture: str = "transformer",
    num_layers: int = 2,
    num_heads: int = 4,
) -> tuple[torch.nn.Module, list[dict]]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.manual_seed(seed)
    layer_count = len(samples[0].layer_ratios)
    model = build_predictor(
        architecture=architecture,
        layer_count=layer_count,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    target = torch.tensor([[sample.global_ratio] for sample in samples], dtype=torch.float32)
    labels = torch.tensor([sample.layer_ratios for sample in samples], dtype=torch.float32)
    weights = torch.tensor([max(sample.reward, 1e-6) for sample in samples], dtype=torch.float32).reshape(-1, 1)
    weights = weights / weights.mean()

    metrics = []
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        predictions = model(target, teacher=labels)
        loss = ((predictions - labels).pow(2) * weights).mean()
        budget_loss = (predictions.mean(dim=1, keepdim=True) - target).pow(2).mean()
        total = loss + budget_loss
        total.backward()
        optimizer.step()
        metrics.append(
            {
                "epoch": epoch,
                "loss": round(float(total.detach()), 8),
                "ratio_loss": round(float(loss.detach()), 8),
                "budget_loss": round(float(budget_loss.detach()), 8),
            }
        )
    return model, metrics

# 用训练好的预测器对给定目标剪枝率推理，输出逐层剪枝率。
# 推理结果经过 budget projection 确保均值等于 target_ratio。
def predict_ratios(model: torch.nn.Module, target_ratio: float) -> tuple[float, ...]:
    with torch.no_grad():
        target = torch.tensor([[target_ratio]], dtype=torch.float32)
        prediction = model(target).reshape(-1).tolist()
    return project_to_budget(tuple(float(value) for value in prediction), target_ratio)

# 将预测的逐层剪枝率投影到满足目标全局剪枝率的约束空间。
# 迭代式调整：每轮找出可调整的层（未触及 min_ratio 或 max_ratio 边界），
# 将差值均匀分摊到可调整层上，直到均值收敛到 target_ratio。
def project_to_budget(
    ratios: tuple[float, ...],
    target_ratio: float,
    min_ratio: float = 0.0,
    max_ratio: float = 0.95,
) -> tuple[float, ...]:
    if not ratios:
        raise ValueError("ratios must not be empty")
    if target_ratio < min_ratio or target_ratio > max_ratio:
        raise ValueError(f"target_ratio must be in [{min_ratio}, {max_ratio}], got {target_ratio}")
    values = [min(max(float(value), min_ratio), max_ratio) for value in ratios]
    target_sum = target_ratio * len(values)
    for _ in range(len(values) * 4):
        current = sum(values)
        diff = target_sum - current
        if abs(diff) < 1e-8:
            return tuple(values)
        if diff > 0:
            adjustable = [index for index, value in enumerate(values) if value < max_ratio]
            if not adjustable:
                break
            step = diff / len(adjustable)
            for index in adjustable:
                values[index] = min(max_ratio, values[index] + step)
        else:
            adjustable = [index for index, value in enumerate(values) if value > min_ratio]
            if not adjustable:
                break
            step = diff / len(adjustable)
            for index in adjustable:
                values[index] = max(min_ratio, values[index] + step)
    current_ratio = sum(values) / len(values)
    if abs(current_ratio - target_ratio) > 1e-6:
        raise ValueError(f"cannot project ratios to target {target_ratio}; got {current_ratio}")
    return tuple(values)

def _validate_layer_count(layer_count: int) -> None:
    if layer_count < 1:
        raise ValueError("layer_count must be positive")
