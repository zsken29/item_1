"""
LOP (Learning Optimal Pruning) 论文复现 —— 单文件独立实现。

论文：LOP: Learning Optimal Pruning for Efficient On-Demand MLLMs Scaling
      https://arxiv.org/abs/2506.12826v1

流水线共 5 步：
  1. 采集 FFN 激活重要性（论文 Appendix A.2，activation L2/RMS）
  2. MCTS 搜索逐层剪枝率（论文 Appendix A.3，UCB + 连续扰动）
  3. 训练 Transformer 自回归预测器（论文 Appendix A.1）
  4. 评测 Dense 模型在 MMBench 上的准确率
  5. 应用 LoP 剪枝并评测剪枝后准确率

运行依赖：Qwen2.5-VL-7B-Instruct 模型 + MMBench dev 标注文件。
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# ── 路径配置 ─────────────────────────────────────────────
MODEL_DIR = ROOT / "models" / "Qwen2.5-VL-7B-Instruct"
MMBENCH_FILE = ROOT / "data" / "mmbench" / "annotations" / "en" / "dev-00000-of-00001.jsonl"

# ── 剪枝超参数（对齐论文 Section 5.1 / Appendix A）─────────
TARGET_RATIO = 0.25          # 最终预测目标：全局 25% FFN 神经元剪枝
TRAIN_RATIOS = [0.20, 0.30, 0.50]  # MCTS 在三个预算下采样，训练预测器泛化到不同剪枝率
SEED = 42
CALIBRATION_SAMPLES = 30     # 重要性采集用的样本数（论文用 500，此处为快速复现降低）

# ── MCTS 参数（论文 Appendix A.3）────────────────────────
MCTS_STEPS = 100             # MCTS 模拟次数（论文用 300）
MCTS_BRANCHES = 4            # 每个节点最多扩展 4 个子节点
MCTS_TOP_CONFIGS = 8         # 每个预算保留 top-8 配置作为训练数据
MCTS_REWARD_LIMIT = 8        # reward 评估样本数

# ── 预测器参数（论文 Appendix A.1.2）──────────────────────
PREDICTOR_EPOCHS = 150       # 训练轮数（论文用 200）
PREDICTOR_HIDDEN_SIZE = 128  # Transformer 隐层维度
PREDICTOR_LAYERS = 2         # Transformer encoder 层数
PREDICTOR_HEADS = 4          # 多头注意力头数

# ── 评测参数 ─────────────────────────────────────────────
EVAL_LIMIT = 100             # MMBench 评测样本数上限（全量为 ~4K）
MAX_NEW_TOKENS = 8           # 生成 token 数（MMBench 只需输出选项字母）
ATTN_IMPLEMENTATION = "sdpa" # PyTorch SDPA 注意力后端，比 eager 更快

# ── 中间产物输出路径 ─────────────────────────────────────
IMPORTANCE_FILE = HERE / "importance.pt"                  # Step 1: FFN 各层神经元重要性
MCTS_FILE = HERE / "mcts_samples.jsonl"                   # Step 2: 全部 MCTS 搜索结果
LOP_TRAINING_FILE = HERE / "lop_training_samples.jsonl"   # Step 2: 精选训练样本（top-8/预算）
LOP_PREDICTOR_FILE = HERE / "lop_predictor.pt"            # Step 3: 训练好的预测器权重
PREDICTION_FILE = HERE / "predicted_ratios.json"           # Step 3: 预测的逐层剪枝率
DENSE_RESULT_FILE = HERE / "dense_mmbench_limit100.jsonl" # Step 4: Dense 模型逐条结果
LOP_RESULT_FILE = HERE / "lop25_mmbench_limit100.jsonl"   # Step 5: LoP 剪枝后逐条结果
SUMMARY_FILE = HERE / "summary.json"                       # 最终汇总


class RatioPredictor(torch.nn.Module):
    """Transformer 自回归剪枝率预测器（论文 Appendix A.1 / Eq. 7-12）。

    设计要点：
    - 输入目标全局剪枝率 b，自回归输出每一层的剪枝率 r_l
    - 第 l 层的预测依赖于 b 和前面 l-1 层的剪枝决策，捕获层间依赖
    - 使用 causal mask 保证自回归特性
    - 输出经 sigmoid 约束到 (0, 1)

    结构：
      budget_embed: 目标剪枝率 -> hidden_size 的 MLP（充当 CLS token）
      layer_embedding: 可学习的位置编码，与每层已预测的剪枝率相乘
      encoder: TransformerEncoder（causal self-attention）
      head: 线性投影 -> sigmoid -> 逐层剪枝率
    """

    def __init__(self, layer_count: int, hidden_size: int = PREDICTOR_HIDDEN_SIZE) -> None:
        super().__init__()
        self.layer_count = layer_count
        self.hidden_size = hidden_size
        # 目标剪枝率编码：1 维标量 -> hidden_size 向量（类似 ViT 的 CLS token）
        self.budget = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, hidden_size),
        )
        # 可学习的位置嵌入：每层一个 hidden_size 向量
        # 初始化使用小方差防止梯度爆炸
        self.layer_embedding = torch.nn.Parameter(torch.randn(layer_count, hidden_size) * 0.02)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=PREDICTOR_HEADS,
            dim_feedforward=hidden_size * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=PREDICTOR_LAYERS)
        self.head = torch.nn.Linear(hidden_size, 1)

    def forward(self, target_ratio: torch.Tensor, teacher: torch.Tensor | None = None) -> torch.Tensor:
        """前向传播。

        Args:
            target_ratio: [B] 目标全局剪枝率
            teacher: [B, L] 可选，训练时提供真实逐层剪枝率做 teacher forcing

        Returns:
            [B, L] 预测的逐层剪枝率
        """
        target_ratio = target_ratio.reshape(-1, 1)
        if teacher is not None:
            # 训练模式：teacher forcing，使用真实历史（非预测值）
            # 在 teacher 前补零作为第 0 层的"之前剪枝率"
            previous = torch.cat([torch.zeros_like(teacher[:, :1]), teacher[:, :-1]], dim=1)
            return self._predict(target_ratio, previous)

        # 推理模式：逐层自回归生成
        previous = torch.zeros(target_ratio.shape[0], self.layer_count, device=target_ratio.device)
        outputs = []
        for layer_index in range(self.layer_count):
            prediction = self._predict(target_ratio, previous)
            current = prediction[:, layer_index : layer_index + 1]
            outputs.append(current)
            # 用预测值（detach 切断梯度）更新历史，供下一层使用
            previous[:, layer_index : layer_index + 1] = current.detach()
        return torch.cat(outputs, dim=1)

    def _predict(self, target_ratio: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        """单步预测：给定目标剪枝率 b 和历史剪枝率，输出所有层剪枝率。

        token 序列: [budget_token, layer_0_token, layer_1_token, ..., layer_{L-1}_token]
        causal mask 确保 layer_i 只能看到 budget_token 和 layer_{0..i}。
        """
        batch_size = target_ratio.shape[0]
        # budget_token: [B, 1, H]
        budget_token = self.budget(target_ratio).unsqueeze(1)
        # layer_tokens: 每层的 previously_predicted_ratio * learnable_embedding
        # 未预测层为 0 -> embedding 不激活，预测层按比例激活
        layer_tokens = previous.unsqueeze(-1) * self.layer_embedding.unsqueeze(0)
        # 拼接成 [B, 1+L, H]
        tokens = torch.cat([budget_token, layer_tokens], dim=1)
        # causal mask: 确保 layer_i 只能 attend budget 和 layer_{0..i}
        mask = torch.nn.Transformer.generate_square_subsequent_mask(tokens.shape[1], device=tokens.device)
        encoded = self.encoder(tokens, mask=mask)
        # 取 layer token 部分（丢弃 budget token），经 sigmoid 映射到 (0,1)
        return torch.sigmoid(self.head(encoded[:, 1:, :])).reshape(batch_size, self.layer_count)


def check_inputs() -> None:
    """验证模型目录和 MMBench 标注文件存在，并确保 CUDA 可用。"""
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"missing model directory: {MODEL_DIR}")
    if not MMBENCH_FILE.is_file():
        raise FileNotFoundError(f"missing MMBench annotation: {MMBENCH_FILE}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Qwen2.5-VL-7B reproduction script")


def print_header(layers: list[dict]) -> None:
    print("Qwen2.5-VL-7B LoP reproduction")
    print(f"  model       : {MODEL_DIR}")
    print(f"  benchmark   : {MMBENCH_FILE}")
    print(f"  layers      : {len(layers)}")
    print(f"  target ratio: {TARGET_RATIO}")
    print(f"  train ratios: {TRAIN_RATIOS}")
    print(f"  outputs     : {HERE}")


def qwen_ffn_layers() -> list[dict]:
    """从 model config 中提取所有 FFN 层的元信息。

    Qwen2.5-VL 的 FFN 位于 model.language_model.layers.{i}.mlp。
    剪枝目标是 mlp.down_proj 的输入激活，即中间神经元的输出。
    每层记录：索引、模块路径、激活 hook 位置、中间层维度。
    """
    config = read_json(MODEL_DIR / "config.json")
    if config["model_type"] != "qwen2_5_vl":
        raise ValueError(f"expected qwen2_5_vl model, got {config['model_type']}")
    layers = []
    for index in range(int(config["num_hidden_layers"])):
        block = f"model.language_model.layers.{index}"
        layers.append(
            {
                "index": index,
                "block": block,
                "activation": f"{block}.mlp.down_proj",
                "intermediate_size": int(config["intermediate_size"]),
            }
        )
    return layers


def load_qwen() -> tuple[torch.nn.Module, object, torch.device]:
    """加载 Qwen2.5-VL-7B-Instruct 模型和处理器。

    强制使用本地文件（local_files_only=True），不依赖网络。
    bfloat16 + SDPA 在保证精度的同时节省显存。
    将 HF 缓存目录指向项目内避免污染全局缓存。
    """
    os.environ["HF_HOME"] = str(HERE / "hf_home")
    os.environ["HF_MODULES_CACHE"] = str(HERE / "hf_home" / "modules")
    os.environ["TRANSFORMERS_CACHE"] = str(HERE / "hf_home" / "transformers")

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from transformers import logging as transformers_logging

    transformers_logging.set_verbosity_error()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=torch.bfloat16,
        attn_implementation=ATTN_IMPLEMENTATION,
        device_map="auto",
        local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR), local_files_only=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    if model.generation_config is not None and model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    return model, processor, model.device


def collect_importance(
    model: torch.nn.Module,
    processor: object,
    device: torch.device,
    layers: list[dict],
    samples: list[dict],
) -> dict[int, torch.Tensor]:
    """Step 1: 采集 FFN 神经元重要性（论文 Appendix A.2, Eq. 24-25）。

    在 mlp.down_proj 的输入处注册 forward pre-hook，
    对每个样本前向传播时记录激活值的 L2 范数（平方和），
    最终返回各层每个神经元的 RMS 激活值作为重要性分数。

    只生成 1 个 token 以触发 FFN 计算，不浪费推理时间。
    """
    sums = {layer["index"]: torch.zeros(layer["intermediate_size"], dtype=torch.float32) for layer in layers}
    counts = {layer["index"]: 0 for layer in layers}
    handles = []

    for layer in layers:
        module = model.get_submodule(layer["activation"])

        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], layer_index: int = layer["index"]) -> None:
            # 激活值形状: [batch*tokens, intermediate_size]
            activation = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
            # 累加平方值（L2 范数的平方），按神经元求和
            sums[layer_index] += (activation * activation).sum(dim=0).cpu()
            counts[layer_index] += activation.shape[0]

        handles.append(module.register_forward_pre_hook(hook))

    with torch.inference_mode():
        for sample in tqdm(samples, desc="activation", unit="sample"):
            generate_answer(model, processor, device, sample, max_new_tokens=1)

    for handle in handles:
        handle.remove()

    # RMS: sqrt(mean(squared_activations))，数值稳定，度量激活量级
    return {index: torch.sqrt(sums[index] / counts[index]).cpu() for index in sums}


def mcts_search(
    layers: list[dict],
    importance: dict[int, torch.Tensor],
    target_ratio: float,
    reward_fn,
) -> list[dict]:
    """Step 2: MCTS 搜索逐层剪枝配置（论文 Appendix A.3, Eq. 26-32）。

    每次迭代：
      1. SELECT: 从根节点沿 UCB 最大路径下降到可扩展节点
      2. EXPAND: 对当前 ratios 施加衰减扰动生成子节点
      3. SIMULATE: 用 reward_fn（masked accuracy）评估该配置
      4. BACKUP: 沿路径回传 reward 更新 value 和 visits

    最终返回所有采样配置，按 reward 降序排列。

    关键设计：
    - 扰动幅度随深度衰减 (delta * 0.9^depth)，越深层搜索越精细
    - 扰动后通过 project_ratios 投影到可行域，保证均值 ≤ target_ratio
    """
    rng = random.Random(SEED)
    tensors = [importance[index] for index in sorted(importance)]
    # 根节点：所有层初始剪枝率 = target_ratio（均匀分配）
    root = {
        "ratios": [target_ratio] * len(tensors),
        "parent": None,
        "children": [],
        "visits": 0,
        "value": 0.0,
        "depth": 0,
    }
    rows = []

    for step in tqdm(range(1, MCTS_STEPS + 1), desc=f"mcts b={target_ratio:.2f}", unit="step"):
        # SELECT: 沿树下降到可扩展节点（子节点数 < MCTS_BRANCHES）
        node = root
        while len(node["children"]) >= MCTS_BRANCHES:
            node = max(node["children"], key=lambda child: ucb(node, child))

        # EXPAND: 对当前节点的 ratios 施加随机扰动，生成子节点
        ratios = perturb_ratios(node["ratios"], target_ratio, node["depth"] + 1, rng)
        child = {"ratios": ratios, "parent": node, "children": [], "visits": 0, "value": 0.0, "depth": node["depth"] + 1}
        node["children"].append(child)

        # SIMULATE: 评估该剪枝配置
        reward = reward_fn(ratios)
        # BACKUP: 沿路径回传 reward
        current = child
        while current is not None:
            current["visits"] += 1
            current["value"] += reward
            current = current["parent"]

        rows.append(
            {
                "target_ratio": target_ratio,
                "layer_ratios": ratios,
                "reward": reward,
                "step": step,
                "importance_retention": importance_retention(tensors, ratios),
                "constraint_ok": mean(ratios) <= target_ratio + 1e-8,
                "layers": len(layers),
            }
        )

    rows.sort(key=lambda row: row["reward"], reverse=True)
    return rows


def ucb(parent: dict, child: dict) -> float:
    """UCB (Upper Confidence Bound) 公式（论文 Eq. 27）。

    Q(s,a) + c_puct * sqrt(log(N_parent) / N_child)
    c_puct = 1.4，平衡 exploration 与 exploitation。
    未访问节点返回 +inf，保证优先探索新节点。
    """
    if child["visits"] == 0:
        return math.inf
    return child["value"] / child["visits"] + 1.4 * math.sqrt(math.log(parent["visits"] + 1) / child["visits"])


def perturb_ratios(ratios: list[float], target_ratio: float, depth: int, rng: random.Random) -> list[float]:
    """对逐层剪枝率施加衰减扰动（论文 Appendix A.3, Eq. 29-30）。

    扰动幅度 delta = 0.1 * 0.9^depth，越深层搜索越精细（局部微调）。
    扰动后投影回可行域 [0.10, 0.95] 并保持均值 ≈ target_ratio。
    下界 0.10 防止某层几乎不剪枝，上界 0.95 防止某层几乎全剪光。
    """
    delta = 0.1 * (0.9 ** depth)
    changed = [value + rng.uniform(-delta, delta) for value in ratios]
    return project_ratios(changed, target_ratio, min_ratio=0.10, max_ratio=0.95)


def importance_retention(importance: list[torch.Tensor], ratios: list[float]) -> float:
    """计算剪枝后的重要性保留率（诊断指标，非 reward）。

    对每层：保留 top-k 高重要性神经元，计算 (保留神经元的重要性之和 / 全部神经元重要性之和)。
    返回各层平均值。越高说明越重要的神经元被保留了。
    """
    retained = []
    for scores, ratio in zip(importance, ratios):
        values = scores.float().reshape(-1)
        keep = max(1, int(round(values.numel() * (1.0 - ratio))))
        top = torch.topk(values, k=keep, largest=True).values.sum()
        retained.append(float(top / values.sum().clamp_min(1e-12)))
    return mean(retained)


def train_predictor(rows: list[dict], layer_count: int) -> tuple[list[float], float, torch.nn.Module]:
    """Step 3: 训练剪枝率预测器（论文 Appendix A.1.2）。

    使用 MCTS 采样得到的 (target_ratio, layer_ratios) 对做监督学习。
    MSE 损失，AdamW 优化器，teacher forcing 训练。
    训练完成后用 TARGET_RATIO 进行一次推理，投影得到最终逐层剪枝率。

    Returns:
        (predicted_ratios, final_loss, model) — 预测的逐层剪枝率、最终损失、训练好的模型
    """
    device = torch.device("cuda")
    model = RatioPredictor(layer_count).to(device)
    targets = torch.tensor([row["target_ratio"] for row in rows], dtype=torch.float32, device=device)
    labels = torch.tensor([row["layer_ratios"] for row in rows], dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    progress = tqdm(range(1, PREDICTOR_EPOCHS + 1), desc="predictor", unit="epoch")
    final_loss = 0.0
    for epoch in progress:
        prediction = model(targets, teacher=labels)
        loss = (prediction - labels).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
        if epoch == 1 or epoch % 25 == 0 or epoch == PREDICTOR_EPOCHS:
            progress.set_postfix(loss=f"{final_loss:.5f}")

    model.eval()
    with torch.no_grad():
        raw = model(torch.tensor([TARGET_RATIO], dtype=torch.float32, device=device)).reshape(-1).cpu().tolist()
    # 推理结果投影：下界 0.0（允许不剪枝），上界 0.95
    return project_ratios([float(value) for value in raw], TARGET_RATIO, min_ratio=0.0, max_ratio=0.95), final_loss, model


def save_predictor(model: torch.nn.Module, final_loss: float, path: Path) -> dict:
    metadata = {
        "architecture": "transformer_autoregressive",
        "layer_count": model.layer_count,
        "hidden_size": model.hidden_size,
        "num_layers": PREDICTOR_LAYERS,
        "num_heads": PREDICTOR_HEADS,
        "parameter_count": parameter_count(model),
        "target_ratio": TARGET_RATIO,
        "train_ratios": TRAIN_RATIOS,
        "mcts_top_configs": MCTS_TOP_CONFIGS,
        "predictor_epochs": PREDICTOR_EPOCHS,
        "final_loss": final_loss,
    }
    torch.save(
        {
            **metadata,
            "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        },
        path,
    )
    return metadata


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def masked_accuracy(
    model: torch.nn.Module,
    processor: object,
    device: torch.device,
    samples: list[dict],
    layers: list[dict],
    importance: dict[int, torch.Tensor],
    ratios: list[float],
) -> float:
    """MCTS reward 函数：安装 FFN mask 后评测准确率（论文 Eq. 31-32）。

    按给定 ratios 对每层保留 top-k 高重要性神经元，其余神经元输出置零。
    mask 通过 forward pre-hook 实现，不修改模型权重，可随时卸载。
    返回准确率作为 MCTS 的 reward 信号。
    """
    handles = install_ffn_masks(model, layers, importance, ratios)
    try:
        correct = 0
        with torch.inference_mode():
            for sample in samples:
                response = generate_answer(model, processor, device, sample, max_new_tokens=MAX_NEW_TOKENS)
                predicted = extract_option(response)
                expected = extract_option(str(sample["fields"]["answer"]))
                correct += int(predicted == expected)
    finally:
        for handle in handles:
            handle.remove()
    return correct / len(samples)


def install_ffn_masks(
    model: torch.nn.Module,
    layers: list[dict],
    importance: dict[int, torch.Tensor],
    ratios: list[float],
) -> list[torch.utils.hooks.RemovableHandle]:
    """为 FFN 中间层安装二进制 mask（模拟剪枝效果，不改变权重）。

    每层根据 ratio 保留 top-k 高重要性神经元，mask 中保留位=1，裁剪位=0。
    mask 通过 forward pre-hook 作用于 mlp.down_proj 的输入激活，
    等价于将对应中间神经元输出置零。

    view_shape 适配任意 batch/token 维度，仅在最后一维（神经元维）做 mask。
    """
    handles = []
    for layer, ratio in zip(layers, ratios):
        module = model.get_submodule(layer["activation"])
        scores = importance[layer["index"]].float().cpu()
        keep_count = max(1, int(round(scores.numel() * (1.0 - ratio))))
        keep = torch.topk(scores, k=keep_count, largest=True, sorted=True).indices
        mask = torch.zeros(scores.numel(), dtype=torch.float32)
        mask[keep] = 1.0

        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], mask: torch.Tensor = mask):
            activation = inputs[0]
            # 广播 mask 到与激活相同维度: [..., intermediate_size]
            view_shape = [1] * (activation.ndim - 1) + [mask.numel()]
            return (activation * mask.to(device=activation.device, dtype=activation.dtype).view(*view_shape),)

        handles.append(module.register_forward_pre_hook(hook))
    return handles


def evaluate_with_masks(
    model: torch.nn.Module,
    processor: object,
    device: torch.device,
    samples: list[dict],
    output_file: Path,
    name: str,
    layers: list[dict],
    importance: dict[int, torch.Tensor],
    ratios: list[float],
) -> dict:
    handles = install_ffn_masks(model, layers, importance, ratios)
    try:
        return evaluate(model, processor, device, samples, output_file, name)
    finally:
        for handle in handles:
            handle.remove()


def pruning_stats(layers: list[dict], ratios: list[float]) -> dict:
    original_total = 0
    kept_total = 0
    for layer, ratio in zip(layers, ratios):
        original = int(layer["intermediate_size"])
        kept = max(1, int(round(original * (1.0 - ratio))))
        original_total += original
        kept_total += kept
    return {
        "original_neurons": original_total,
        "kept_neurons": kept_total,
        "actual_ratio": 1.0 - kept_total / original_total,
        "application": "mask_eval_due_to_device_map_offload",
    }


def has_meta_parameters(model: torch.nn.Module) -> bool:
    """检测模型是否使用了 device_map="auto" 导致的 meta 设备参数。

    当模型跨多 GPU 或使用 disk offload 时，部分参数在加载前位于 meta 设备。
    meta 设备上的参数没有真实数据，无法直接做 weight copy 剪枝，
    此时只能用 mask 模拟。
    """
    return any(parameter.device.type == "meta" for parameter in model.parameters())


def prune_ffn(
    model: torch.nn.Module,
    layers: list[dict],
    importance: dict[int, torch.Tensor],
    ratios: list[float],
) -> dict:
    """Step 5 (真实剪枝路径): 根据逐层剪枝率物理删除 FFN 中间神经元。

    对每层 MLP 的三个投影矩阵做结构性裁剪（论文 Section 3.1, Eq. 1）：
    - gate_proj: 保留 keep 行（输出维度裁剪）
    - up_proj:   保留 keep 行（输出维度裁剪）
    - down_proj: 保留 keep 列（输入维度裁剪）
    - 更新 intermediate_size 确保后续 forward shape 一致

    返回剪枝前后神经元总数和实际剪枝率。
    """
    original_total = 0
    kept_total = 0

    for layer, ratio in zip(layers, ratios):
        mlp = model.get_submodule(layer["block"]).mlp
        gate = mlp.gate_proj
        up = mlp.up_proj
        down = mlp.down_proj
        scores = importance[layer["index"]].to(gate.weight.device)
        keep_count = max(1, int(round(scores.numel() * (1.0 - ratio))))
        keep = torch.topk(scores, k=keep_count, largest=True, sorted=True).indices.sort().values.to(torch.long)

        # gate/up: 行裁剪（输出神经元减少）
        mlp.gate_proj = keep_linear_rows(gate, keep)
        mlp.up_proj = keep_linear_rows(up, keep)
        # down: 列裁剪（输入神经元减少，与 gate/up 的输出维度一致）
        mlp.down_proj = keep_linear_columns(down, keep)
        if hasattr(mlp, "intermediate_size"):
            mlp.intermediate_size = keep_count

        original_total += int(scores.numel())
        kept_total += keep_count

    return {
        "original_neurons": original_total,
        "kept_neurons": kept_total,
        "actual_ratio": 1.0 - kept_total / original_total,
    }


def keep_linear_rows(linear: torch.nn.Linear, keep: torch.Tensor) -> torch.nn.Linear:
    """按行裁剪 Linear 层：保留 keep 指定的输出神经元。

    用于 gate_proj 和 up_proj（输出维度 = intermediate_size）。
    新层 out_features = keep.numel()，in_features 不变。
    """
    new_linear = torch.nn.Linear(
        linear.in_features,
        int(keep.numel()),
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    with torch.no_grad():
        new_linear.weight.copy_(linear.weight.index_select(0, keep))
        if linear.bias is not None:
            new_linear.bias.copy_(linear.bias.index_select(0, keep))
    return new_linear


def keep_linear_columns(linear: torch.nn.Linear, keep: torch.Tensor) -> torch.nn.Linear:
    """按列裁剪 Linear 层：保留 keep 指定的输入神经元。

    用于 down_proj（输入维度 = intermediate_size）。
    新层 in_features = keep.numel()，out_features 不变。
    bias 不需要裁剪（bias 对应输出维度，不在被裁剪的维度上）。
    """
    new_linear = torch.nn.Linear(
        int(keep.numel()),
        linear.out_features,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    with torch.no_grad():
        new_linear.weight.copy_(linear.weight.index_select(1, keep))
        if linear.bias is not None:
            new_linear.bias.copy_(linear.bias)
    return new_linear


def evaluate(
    model: torch.nn.Module,
    processor: object,
    device: torch.device,
    samples: list[dict],
    output_file: Path,
    name: str,
) -> dict:
    correct = 0
    with output_file.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for sample in tqdm(samples, desc=f"eval {name}", unit="sample"):
            response = generate_answer(model, processor, device, sample, max_new_tokens=MAX_NEW_TOKENS)
            predicted = extract_option(response)
            expected = extract_option(str(sample["fields"]["answer"]))
            row = {
                "source_row": sample["source_row"],
                "predicted": predicted,
                "expected": expected,
                "correct": predicted == expected,
                "response": response,
            }
            correct += int(row["correct"])
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {"accuracy": correct / len(samples), "correct": correct, "total": len(samples)}


def generate_answer(
    model: torch.nn.Module,
    processor: object,
    device: torch.device,
    sample: dict,
    max_new_tokens: int,
) -> str:
    """用 Qwen2.5-VL 模型生成单个样本的回答。

    构建图文多模态消息（1 张图片 + 选项问题），
    使用 greedy decoding（do_sample=False, num_beams=1）保证确定性输出。
    """
    from qwen_vl_utils import process_vision_info

    image_path = ROOT / sample["images"][0]["path"]
    if not image_path.is_file():
        raise FileNotFoundError(f"missing image: {image_path}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": build_question(sample)},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
    # 截取生成部分（去掉输入 token）
    output_ids = [output[len(input_ids) :] for input_ids, output in zip(inputs.input_ids, output_ids)]
    return processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def build_question(sample: dict) -> str:
    """构建 MMBench 多选题的 prompt 文本。

    格式: [hint] + question + A. xxx / B. xxx / C. xxx / D. xxx + 指令
    结尾指令 "Answer with the option letter only." 让模型仅输出选项字母。
    """
    fields = sample["fields"]
    parts = []
    hint = str(fields.get("hint", "nan"))
    if hint and hint != "nan":
        parts.append(hint)
    parts.append(str(fields["question"]))
    for option in ("A", "B", "C", "D"):
        value = str(fields.get(option, "nan"))
        if value != "nan":
            parts.append(f"{option}. {value}")
    parts.append("Answer with the option letter only.")
    return "\n".join(parts)


def extract_option(text: str) -> str:
    """从模型输出中提取选项字母 A/B/C/D。

    优先匹配独立的 A-D 字母（\b 单词边界），
    回退：取规范化后文本的首字符。
    """
    match = re.search(r"\b([A-D])\b", text.upper())
    if match:
        return match.group(1)
    normalized = re.sub(r"\s+", " ", text.strip().upper().strip(".,;:!?'\"")).strip()
    return normalized[:1]


def calibration_samples() -> list[dict]:
    samples = read_samples()
    rng = random.Random(SEED)
    indices = sorted(rng.sample(range(len(samples)), CALIBRATION_SAMPLES))
    return [samples[index] for index in indices]


def read_samples(limit: int | None = None) -> list[dict]:
    rows = []
    with MMBENCH_FILE.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            for key in ("source_row", "fields", "images"):
                if key not in row:
                    raise ValueError(f"{MMBENCH_FILE}:{line_number}: missing {key}")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def project_ratios(values: list[float], target: float, min_ratio: float, max_ratio: float) -> list[float]:
    """将逐层剪枝率投影到可行域并使均值逼近 target。

    两步：先 clip 到 [min_ratio, max_ratio]，再迭代均匀调整可调维度。
    "可调"指 diff>0 时未达上界的维度，或 diff<0 时未达下界的维度。
    最多 100 轮迭代，tol=1e-8 时收敛。
    """
    ratios = [min(max_ratio, max(min_ratio, value)) for value in values]
    for _ in range(100):
        diff = target - mean(ratios)
        if abs(diff) < 1e-8:
            break
        adjustable = [
            index
            for index, value in enumerate(ratios)
            if (diff > 0 and value < max_ratio - 1e-10) or (diff < 0 and value > min_ratio + 1e-10)
        ]
        if not adjustable:
            break
        step = diff * len(ratios) / len(adjustable)
        for index in adjustable:
            ratios[index] = min(max_ratio, max(min_ratio, ratios[index] + step))
    return ratios


def print_summary(name: str, summary: dict, output_file: Path) -> None:
    print(f"      accuracy={summary['accuracy']:.4f} ({summary['correct']}/{summary['total']})")
    print(f"      saved: {output_file}")


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    """LOP 复现主流水线（论文 Figure 3 整体流程）。

    Step 1 ─ 采集 FFN 激活重要性
      - 在校准集上运行模型，hook mlp.down_proj 输入
      - 计算每个神经元的 RMS 激活值作为重要性分数

    Step 2 ─ MCTS 搜索逐层剪枝配置
      - 在 {0.20, 0.30, 0.50} 三个预算下各搜索 100 步
      - reward = 安装 mask 后在 reward 样本上的准确率
      - top-8 配置作为预测器训练数据

    Step 3 ─ 训练剪枝率预测器
      - Transformer 自回归模型，输入 target_ratio → 输出逐层 ratios
      - 训练后预测 TARGET_RATIO=0.25 对应的逐层剪枝率

    Step 4 ─ Dense 基线评测
      - 未剪枝模型在 EVAL_LIMIT 个 MMBench 样本上的准确率

    Step 5 ─ LoP 剪枝 + 评测
      - 若模型有 meta 参数（device_map 卸载），用 mask 模拟
      - 否则真实裁剪 FFN 权重矩阵
      - 评测剪枝后准确率
    """
    started = time.time()
    random.seed(SEED)
    torch.manual_seed(SEED)
    check_inputs()

    layers = qwen_ffn_layers()
    print_header(layers)

    # ── Step 1: 采集 FFN 激活重要性 ──
    print("\n[1/5] Collect FFN activation importance")
    model, processor, device = load_qwen()
    calibration = calibration_samples()
    importance = collect_importance(model, processor, device, layers, calibration)
    torch.save({"target": "mlp.down_proj input RMS", "importance": importance}, IMPORTANCE_FILE)
    print(f"      saved: {IMPORTANCE_FILE}")

    # ── Step 2: MCTS 搜索 ──
    print("\n[2/5] Search layer pruning ratios with MCTS")
    reward_samples = read_samples(MCTS_REWARD_LIMIT)
    mcts_rows = []
    training_rows = []
    for budget in TRAIN_RATIOS:
        rows = mcts_search(
            layers,
            importance,
            budget,
            # reward_fn: 用当前配置做 mask，评测准确率
            lambda ratios, budget=budget: masked_accuracy(
                model,
                processor,
                device,
                reward_samples,
                layers,
                importance,
                ratios,
            ),
        )
        mcts_rows.extend(rows)
        # 每个预算取 top-k 配置作为训练数据，使预测器学会泛化到不同剪枝率
        training_rows.extend(rows[:MCTS_TOP_CONFIGS])
    del model, processor
    torch.cuda.empty_cache()
    write_jsonl(MCTS_FILE, mcts_rows)
    write_jsonl(LOP_TRAINING_FILE, training_rows)
    best = max(mcts_rows, key=lambda row: row["reward"])
    print(f"      best_reward={best['reward']:.6f}, mean_ratio={mean(best['layer_ratios']):.4f}")
    print(f"      train_configs={len(training_rows)}, reward_samples={MCTS_REWARD_LIMIT}")
    print(f"      saved: {MCTS_FILE}")
    print(f"      saved training data: {LOP_TRAINING_FILE}")

    # ── Step 3: 训练预测器 ──
    print("\n[3/5] Train pruning-ratio predictor")
    predicted_ratios, predictor_loss, predictor_model = train_predictor(training_rows, len(layers))
    predictor_metadata = save_predictor(predictor_model, predictor_loss, LOP_PREDICTOR_FILE)
    del predictor_model
    torch.cuda.empty_cache()
    write_json(
        PREDICTION_FILE,
        {
            "target_ratio": TARGET_RATIO,
            "mean_ratio": mean(predicted_ratios),
            "layer_ratios": predicted_ratios,
            "final_loss": predictor_loss,
            "predictor_parameters": predictor_metadata["parameter_count"],
        },
    )
    print(f"      final_loss={predictor_loss:.6f}, predicted_mean_ratio={mean(predicted_ratios):.4f}")
    print(f"      predictor_parameters={predictor_metadata['parameter_count']}")
    print(f"      saved checkpoint: {LOP_PREDICTOR_FILE}")
    print(f"      saved: {PREDICTION_FILE}")

    samples = read_samples(EVAL_LIMIT)

    # ── Step 4: Dense 基线 ──
    print("\n[4/5] Run dense Qwen2.5-VL-7B on MMBench")
    model, processor, device = load_qwen()
    dense_summary = evaluate(model, processor, device, samples, DENSE_RESULT_FILE, "dense")
    del model, processor
    torch.cuda.empty_cache()
    print_summary("dense", dense_summary, DENSE_RESULT_FILE)

    # ── Step 5: LoP 剪枝 + 评测 ──
    print("\n[5/5] Apply LoP pruning and run MMBench")
    model, processor, device = load_qwen()
    if has_meta_parameters(model):
        # device_map="auto" 可能导致参数在 meta 设备上，无法物理裁剪
        # 此时用 mask hook 模拟剪枝效果
        pruning_summary = pruning_stats(layers, predicted_ratios)
        lop_summary = evaluate_with_masks(
            model,
            processor,
            device,
            samples,
            LOP_RESULT_FILE,
            "lop25",
            layers,
            importance,
            predicted_ratios,
        )
    else:
        # 所有参数在同一设备 → 物理删除神经元，真正减小模型
        pruning_summary = prune_ffn(model, layers, importance, predicted_ratios)
        lop_summary = evaluate(model, processor, device, samples, LOP_RESULT_FILE, "lop25")
    del model, processor
    torch.cuda.empty_cache()
    print_summary("lop25", lop_summary, LOP_RESULT_FILE)

    # ── 汇总输出 ──
    summary = {
        "model": MODEL_DIR.name,
        "dataset": "MMBench dev en",
        "target_ratio": TARGET_RATIO,
        "calibration_samples": CALIBRATION_SAMPLES,
        "train_ratios": TRAIN_RATIOS,
        "mcts_steps": MCTS_STEPS,
        "mcts_branches": MCTS_BRANCHES,
        "mcts_top_configs": MCTS_TOP_CONFIGS,
        "mcts_reward_limit": MCTS_REWARD_LIMIT,
        "predictor_epochs": PREDICTOR_EPOCHS,
        "predictor": predictor_metadata,
        "eval_limit": EVAL_LIMIT,
        "dense": dense_summary,
        "lop25": lop_summary,
        "pruning": pruning_summary,
        "files": {
            "importance": str(IMPORTANCE_FILE),
            "mcts": str(MCTS_FILE),
            "lop_training_samples": str(LOP_TRAINING_FILE),
            "lop_predictor": str(LOP_PREDICTOR_FILE),
            "prediction": str(PREDICTION_FILE),
            "dense_rows": str(DENSE_RESULT_FILE),
            "lop25_rows": str(LOP_RESULT_FILE),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(SUMMARY_FILE, summary)

    print("\nDone")
    print(f"  dense accuracy : {dense_summary['accuracy']:.4f} ({dense_summary['correct']}/{dense_summary['total']})")
    print(f"  lop25 accuracy : {lop_summary['accuracy']:.4f} ({lop_summary['correct']}/{lop_summary['total']})")
    print(f"  actual pruning : {pruning_summary['actual_ratio']:.4f}")
    print(f"  summary        : {SUMMARY_FILE}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
