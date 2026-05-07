"""
qwen2_5_vl.py

一个忠于原版 Qwen2.5-VL 的教学型架构模拟器。

本文件正确建模的部分：
- 类 Qwen2.5-VL-7B 的配置参数
- 视觉编码器：Conv3D  patch 嵌入、RMSNorm、SwiGLU MLP、
  窗口/全局注意调度、视觉 RoPE、PatchMerger
- 文本解码器：Qwen 风格 decoder-only Transformer、RMSNorm、SwiGLU、GQA、
  多模态 RoPE 分段
- 通过图像/视频占位符 token 注入视觉 token（而非简单前插）

本文件故意未实现的部分：
- 真实的 Qwen2.5-VL 权重
- Hugging Face 打包视觉输入格式
- FlashAttention / cu_seqlens 内核
- 官方视觉编码器中使用的精确 window_index 重排
- KV cache / 生产级生成

本文件用于架构学习、模块级剪枝原型设计和形状追踪。
实际实验请使用：
    transformers.Qwen2_5_VLForConditionalGeneration
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 配置
# ============================================================

@dataclass
class ModelConfig:
    """类 Qwen2.5-VL-7B 的架构参数。

    默认值在架构层面与公开的 Qwen2.5-VL-7B-Instruct 配置一致。
    底部的 demo 会覆盖为更小的值以便在普通硬件上运行。
    """

    # ==================== 视觉编码器配置 ====================
    patch_size: int = 14                           # 每个 patch 的空间尺寸（像素）
                                                      # 例如 224x224 图像用 14x14 patch = 16x16 = 256 patches
    temporal_patch_size: int = 2                    # 时间维 patch 大小（视频用）
                                                      # 每 2 帧组成一个时间 patch
    spatial_merge_size: int = 2                    # PatchMerger 中 2×2 相邻 patch 合并
                                                      # 4 个相邻 patch 合并成 1 个，大幅减少 token 数量
    vision_hidden_dim: int = 1280                   # ViT hidden dimension
                                                      # 每 patch 的特征维度
    vision_mlp_hidden_dim: int = 3420              # ViT SwiGLU 中间层维度
                                                      # 约为 vision_hidden_dim 的 2.67 倍
    num_vision_blocks: int = 32                    # Vision Transformer 层数
    vision_num_heads: int = 16                      # ViT 注意力头数
                                                      # head_dim = 1280 / 16 = 80
    vision_window_size: int = 112                   # Window Attention 窗口大小
                                                      # 控制每个 token 可见的局部范围，降低计算量
    full_attention_block_indices: Tuple[int, ...] = (7, 15, 23, 31)
                                                      # 仅这 4 层使用全局注意力（全序列可见）
                                                      # 其余层使用窗口注意力

    # ==================== LLM 解码器配置 ====================
    llm_hidden_dim: int = 3584                     # LLM hidden dimension
                                                      # Qwen2.5-7B 的主要特征维度
    llm_mlp_hidden_dim: int = 18944               # LLM SwiGLU 中间层维度
                                                      # 约为 llm_hidden_dim 的 5.3 倍（比 ViT 大很多）
    num_llm_blocks: int = 28                      # LLM Decoder 层数
    llm_num_attention_heads: int = 28              # LLM 注意力头数（Query 头数）
                                                      # Grouped Query Attention：28 个 query heads
    llm_num_key_value_heads: int = 4              # Key/Value 头数
                                                      # GQA 关键：只有 4 个 KV heads，远少于 28 个 Q heads
                                                      # 大幅减少 KV 缓存和计算量
    vocab_size: int = 152064                        # 词表大小（官方 152064）
    max_position_embeddings: int = 128000           # 最大位置编码长度

    # ==================== RoPE / MRoPE 配置 ====================
    rope_theta: float = 1_000_000.0                 # LLM RoPE theta
                                                      # 官方使用 1000000.0，远大于常用值 10000.0
    vision_rope_theta: float = 10_000.0            # ViT RoPE theta
                                                      # ViT 用更小的 theta，位置信息衰减更快
    mrope_section: Tuple[int, int, int] = (16, 24, 24)
                                                      # MRoPE 三维分段
                                                      # head_dim 被分成三部分：
                                                      # 前 16 维 = temporal（时间）
                                                      # 中 24 维 = height（高度）
                                                      # 后 24 维 = width（宽度）
                                                      # 总计 64 维 = 1280 / 20（当 head_dim=64 时）

    # ==================== 特殊 token id ====================
    # 这些 ID 对应 Qwen2.5-VL tokenizer 中的特殊 token
    bos_token_id: int = 151643                      # BOS 开始 token（<s>）
    eos_token_id: int = 151645                     # EOS 结束 token（</s>）
    vision_start_token_id: int = 151652            # <|vision_start|> 视觉开始标记
    vision_end_token_id: int = 151653               # <|vision_end|> 视觉结束标记
    vision_token_id: int = 151654                  # 视觉占位符基础 token
    image_token_id: int = 151655                   # 图像 token（<|image_pad|>）
                                                      # 图像模式下用于填充位置
    video_token_id: int = 151656                   # 视频 token（<|video_pad|>）
                                                      # 视频模式下用于填充位置

    # ==================== 其他配置 ====================
    rms_norm_eps: float = 1e-6                     # RMSNorm epsilon
                                                      # 防止除零的小常数
    initializer_range: float = 0.02                # 权重初始化标准差
    tie_word_embeddings: bool = False              # 是否共享 embedding
                                                      # True = lm_head 和 embed_tokens 共享权重


# ============================================================
# 基础工具函数
# ============================================================

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    RoPE rotate-half 操作：将向量分成两半，右半部分取负后拼接。

    【数学原理】
    标准 RoPE 对第 pos 个位置的 token，其 query/key 向量旋转角度为 pos × θ。
    对于 2D 向量 [x1, x2]，旋转矩阵是：
        [cos(θ)  -sin(θ)]
        [sin(θ)   cos(θ)] × [x1]
                            [x2]

    【实现技巧】
    将向量分成两半 x1, x2，返回 [-x2, x1]。
    与 cos/sin 相乘后：
        x1 * cos + (-x2) * sin = x1*cos - x2*sin（等于旋转后的第一个分量）
        x2 * cos + x1 * sin = x2*cos + x1*sin（等于旋转后的第二个分量）
    这正好等价于乘以旋转矩阵！

    【为什么这样做】
    - 不需要构造复数或三角函数
    - 只需一次 cat 操作即可完成旋转
    - 计算效率高
    """
    x1 = x[..., : x.shape[-1] // 2]                # 前半部分
    x2 = x[..., x.shape[-1] // 2 :]                # 后半部分
    return torch.cat((-x2, x1), dim=-1)             # [-x2, x1]


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    GQA（Grouped Query Attention）中重复 K/V 头。

    【背景】
    标准 Multi-Head Attention 中，Q/K/V 头数相同。
    Grouped Query Attention (GQA) 让 K/V 头数少于 Q 头数。

    Qwen2.5-7B 配置：
    - num_attention_heads (Q) = 28
    - num_key_value_heads (K/V) = 4
    - n_rep = 28 / 4 = 7

    【为什么这样做】
    1. 减少 KV 缓存：只需要存 4 个头的 KV，而非 28 个
    2. 减少计算量：K/V 投影只需计算 4 个头
    3. 保持多头结构：Q 有 28 个头，每个 head 独立

    【具体操作】
    输入:  [B, 4, N, 80]  (4 个 KV heads)
    扩展后: [B, 28, N, 80] (复制 7 份)

    实现用 expand + reshape，无额外计算成本（共享内存）。
    """
    if n_rep == 1:
        return hidden_states
    bsz, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        bsz, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(bsz, num_kv_heads * n_rep, seq_len, head_dim)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    应用标准 RoPE（旋转位置编码）到 query 和 key。

    【核心公式】
    RoPE(x, pos) = x * cos(pos × θ) + rotate_half(x) * sin(pos × θ)

    其中 cos/sin 的维度是 head_dim // 2，
    因为 rotate_half 把向量分成两半，每半共享同一个角度值。

    【参数】
    q/k: [B, heads, N, head_dim]  query 和 key
    cos/sin: 可广播到 [B, heads, N, head_dim]

    【处理流程】
    1. 如果 cos/sin 是 3D（只给了 [N, dim]），unsqueeze 到 [1, 1, N, dim]
    2. 应用公式：q * cos + rotate_half(q) * sin
    """
    if cos.ndim == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_multimodal_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    应用 Qwen 风格 MRoPE（多模态 RoPE）分段编码。

    【与标准 RoPE 的区别】
    标准 RoPE：所有维度共用同一个位置编码（一维）
    MRoPE：把 head_dim 分成三段，分别编码时间、高度、宽度

    【为什么需要 MRoPE】
    1. 视觉 token 有空间结构：一维位置无法表达 2D 网格位置
    2. 视频 token 还有时间维度
    3. 动态分辨率下，不同图像的 token 数量不同

    【分段机制 mrope_section = (16, 24, 24)】
    head_dim = 64 时（实际是 3584/28=128，但分段按 head_dim//2 算）
    - 前 16 维 × 2 = 32 维 → temporal（时间）
    - 中 24 维 × 2 = 48 维 → height（高度）
    - 后 24 维 × 2 = 48 维 → width（宽度）

    【实现步骤】
    1. mrope_section × 2：变成 [32, 48, 48]，因为 cos/sin 是 duplicated
    2. cos.split：按 [32, 48, 48] 切分 cos → 3 个 chunks
    3. 交错取用：chunk[0] 用 temporal 表，chunk[1] 用 height 表，chunk[2] 用 width 表
       （通过 chunk[i % 3] 实现）
    4. 拼成完整向量

    【文本 token 的情况】
    文本 token 的 position_ids 三个维度值相同（t=h=w），
    所以三个表的频率相同，MROPE 退化为标准 1D RoPE。
    """
    # ×2 因为每个 RoPE 频率在 cos/sin 中出现两次（dupplicated）
    doubled_sections = [s * 2 for s in mrope_section]
    if sum(doubled_sections) != q.shape[-1]:
        # 小型 demo 配置的回退处理：平均分成三段
        base = q.shape[-1] // 3
        doubled_sections = [base, base, q.shape[-1] - 2 * base]

    # 按分段切分 cos/sin
    cos_chunks = cos.split(doubled_sections, dim=-1)
    sin_chunks = sin.split(doubled_sections, dim=-1)

    # 交错取用：按顺序从 temporal/height/width 表中取 chunk
    # chunk[0] → temporal 表，chunk[1] → height 表，chunk[2] → width 表
    # chunk[3] → 又回到 temporal 表（应对 head_dim 无法整除 3 的情况）
    cos_selected = torch.cat(
        [chunk[i % 3] for i, chunk in enumerate(cos_chunks)], dim=-1
    ).unsqueeze(1)
    sin_selected = torch.cat(
        [chunk[i % 3] for i, chunk in enumerate(sin_chunks)], dim=-1
    ).unsqueeze(1)

    q_embed = (q * cos_selected) + (rotate_half(q) * sin_selected)
    k_embed = (k * cos_selected) + (rotate_half(k) * sin_selected)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """
    SwiGLU 激活函数（Qwen/LLaMA 风格的门控 MLP）。

    【来源】
    GLU（Gated Linear Unit）：output = W1 × x * σ(W2 × x)
    其中 σ 是 sigmoid，* 是逐元素乘法

    【SwiGLU 的改进】
    将激活函数换成 SiLU（aka Swish）：output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    其中 SiLU(x) = x * sigmoid(x)

    【三个投影矩阵的作用】
    - gate_proj(x)：产生门控信号，决定哪些信息应该通过
    - up_proj(x)：产生主要信号
    - down_proj(x)：汇总并降维回 hidden_dim

    【为什么用 SwiGLU】
    1. 门控机制让模型学会选择性地激活神经元
    2. SiLU 比 ReLU 更平滑，梯度流动更好
    3. LLaMA/Qwen 等主流模型都用这个
    """
    def __init__(self, hidden_dim: int, intermediate_dim: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)  # 门控投影
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)    # 上投影
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=bias)   # 下投影

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ============================================================
# RoPE 模块
# ============================================================

class VisionRotaryEmbedding(nn.Module):
    """
    ViT 侧使用的视觉 RoPE。

    【设计思路】
    官方视觉编码器为每个 patch 位置构建 2D h/w 旋转表，
    然后复制以匹配注意力头维度。

    【关键特点】
    1. 只在 height 和 width 两个维度构建位置表（无时间维）
    2. 时间维度通过在每帧中复用相同的 h/w 表实现
    3. 使用独立的 vision_rope_theta = 10000.0（比 LLM 的 1000000.0 小很多）

    【数学】
    inv_freq[i] = 1 / θ^(2i/d)，i = 0, 1, ..., d/2-1
    这个公式生成按指数衰减的频率，用于编码不同粒度的位置信息。
    """
    def __init__(self, dim: int, theta: float = 10_000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        # inv_freq[i] = 1 / θ^(2i/d)
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        """返回 [seqlen, dim] 的位置编码表。"""
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


class TextRotaryEmbedding(nn.Module):
    """
    文本侧旋转位置编码，返回三个 MRoPE 表。

    【核心输出】
    生成 [3, B, N, head_dim] 形状的 cos/sin：
    - dim 0：temporal（时间）位置编码
    - dim 1：height（高度）位置编码
    - dim 2：width（宽度）位置编码

    【与 VisionRotaryEmbedding 的区别】
    - VisionRotaryEmbedding：返回 2D h/w 表，每帧复用
    - TextRotaryEmbedding：返回 3 个独立的表，对应 t/h/w 三个维度

    【为什么需要三个表】
    因为 MRoPE 需要分别编码时间、高度、宽度。
    这三个维度在数学上是独立的，需要独立的位置信息。
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        head_dim = config.llm_hidden_dim // config.llm_num_attention_heads
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            position_ids: [3, B, N]
                - position_ids[0] = temporal 位置 ID
                - position_ids[1] = height 位置 ID
                - position_ids[2] = width 位置 ID

        返回:
            cos/sin: [3, B, N, head_dim]
                - dim 0 = temporal cos/sin
                - dim 1 = height cos/sin
                - dim 2 = width cos/sin
        """
        if position_ids.ndim != 3 or position_ids.shape[0] != 3:
            raise ValueError("position_ids must have shape [3, batch, seq_len].")

        # [3, B, N, head_dim/2] = position_ids * inv_freq
        freqs = position_ids.to(self.inv_freq.dtype).unsqueeze(-1) * self.inv_freq.view(1, 1, 1, -1)

        # [3, B, N, head_dim] — duplicated 复制到完整维度
        # 前 half 和后 half 相同，实现 sin/cos 的周期性
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        return cos, sin


# ============================================================
# Attention 实现
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    MHA/GQA 注意力模块，支持可选的 causal masking 和可选的 RoPE。

    【支持的两种模式】
    1. 视觉 MHA: num_kv_heads == num_heads, is_causal=False
       - 所有头独立计算
       - 双向注意力（所有 token 互相可见）

    2. LLM GQA: num_kv_heads < num_heads, is_causal=True
       - 4 个 KV heads → 28 个 Q heads（通过 repeat_kv 扩展）
       - 因果注意力（只能看前面的 token）

    【完整前向流程】
    1. Q/K/V 投影：hidden_dim → num_heads × head_dim
    2. （可选）应用 RoPE/MRoPE 到 Q 和 K
    3. （GQA 专用）重复 K/V 头以匹配 Q 头数
    4. 计算注意力分数：QK^T / sqrt(d_k)
    5. 应用 mask（causal / window / 全 1）
    6. Softmax + 加权求和
    7. 输出投影：num_heads × head_dim → hidden_dim
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        is_causal: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.num_kv_groups = num_heads // self.num_kv_heads  # GQA 中每组有多少 Q heads
        self.head_dim = hidden_dim // num_heads
        self.is_causal = is_causal
        self.scaling = self.head_dim ** -0.5  # 1/sqrt(d_k)

        # Q 投影：hidden_dim → num_heads × head_dim
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=bias)
        # K/V 投影：hidden_dim → num_kv_heads × head_dim（更窄）
        self.k_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=bias)
        # O 投影：num_heads × head_dim → hidden_dim
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_dim, bias=False)

    def _causal_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        创建 causal mask（下三角 mask）。

        【目的】
        保证 Decoder 中第 t 个 token 只能看到位置 0...t-1 的信息，
        不能看到未来位置 t, t+1, ...（否则是信息泄漏）。

        【返回值】
        True = 可见（可以 attend 到）
        False = 被 mask（不能 attend）

        【实现逻辑】
        torch.tril 生成下三角矩阵：
        [[1,0,0],
         [1,1,0],
         [1,1,1]]
        对角线以下为 1（有效），以上为 0（mask）

        然后取反：True = 需要被 mask
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        mask = mask.view(1, 1, seq_len, seq_len).expand(batch_size, 1, seq_len, seq_len)

        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError("attention_mask must have shape [B, N].")
            # attention_mask: [B, N]，1 = 有效，0 = 无效
            key_mask = attention_mask.to(torch.bool).view(batch_size, 1, 1, seq_len)
            mask = mask & key_mask

        return mask

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mrope_section: Optional[Tuple[int, int, int]] = None,
        custom_visibility_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        # ========== 投影 + reshape 到多头格式 ==========
        # [B, N, hidden_dim] → [B, N, num_heads, head_dim] → [B, num_heads, N, head_dim]
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # ========== 应用 RoPE/MRoPE ==========
        if position_embeddings is not None:
            cos, sin = position_embeddings
            if mrope_section is None:
                # 标准 RoPE（视觉用）
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
            else:
                # MRoPE（LLM 文本用）
                q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)

        # ========== GQA：重复 K/V 头 ==========
        # 将 [B, 4, N, 80] 扩展成 [B, 28, N, 80]
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        # ========== 计算注意力分数 ==========
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # ========== 确定可见性 mask ==========
        # 优先级：custom_visibility_mask > causal_mask > 全 1 mask
        if custom_visibility_mask is not None:
            # 自定义可见性掩码（用于 Window Attention）
            visible = custom_visibility_mask.to(torch.bool)
        elif self.is_causal:
            # 因果掩码（LLM 用）
            visible = self._causal_mask(bsz, seq_len, x.device, attention_mask)
        else:
            # 全 1 掩码（视觉 MHA 用，双向注意力）
            visible = torch.ones(bsz, 1, seq_len, seq_len, device=x.device, dtype=torch.bool)

        # ========== Mask + Softmax ==========
        # ~visible = True 表示该位置需要被 mask 掉
        scores = scores.masked_fill(~visible, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # ========== 合并多头 + 输出投影 ==========
        out = out.transpose(1, 2).reshape(bsz, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(out)


# ============================================================
# 视觉侧模块
# ============================================================

class VisionPatchEmbed(nn.Module):
    """
    图像/视频的 Conv patch 嵌入层。

    【功能】
    将输入图像/视频转换为 patch tokens 序列。

    【输入形式】
    - 图像：image [B, 3, H, W]
    - 视频：video [B, 3, T, H, W]

    【输出形式】
    - embeddings: [B, num_patches, embed_dim]
    - grid_thw: (T', H', W') — patch 网格尺寸

    【关键设计】
    使用 3D 卷积同时在时间、高度、宽度三个维度做 patch 划分。
    - 对于图像（T=1）：等价于 2D patch embedding
    - 对于视频（T>1）：每 temporal_patch_size 帧组成一个时间 patch

    【数值例子】
    224×224 图像，patch_size=14 → 16×16 = 256 patches
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.embed_dim = config.vision_hidden_dim

        # 视频：Conv3D，kernel_size = (temporal, spatial, spatial)
        # 把 T 帧分成 T/temporal_patch_size 组，每组空间上用 14x14 划分
        self.proj_video = nn.Conv3d(
            3,                                              # RGB 3 通道
            self.embed_dim,                                 # 输出 embed_dim 维 token
            kernel_size=(
                config.temporal_patch_size,                # 时间维：每 temporal_patch_size 帧一组
                config.patch_size,                          # 空间维：14x14
                config.patch_size,
            ),
            stride=(
                config.temporal_patch_size,
                config.patch_size,
                config.patch_size,
            ),
            bias=False,                                    # 无 bias（现代 ViT 设计）
        )

        # 图像：Conv2D（更直观，避免 T=1 时 temporal_patch_size>1 的问题）
        self.proj_image = nn.Conv2d(
            3,
            self.embed_dim,
            kernel_size=config.patch_size,                   # 14x14 patch
            stride=config.patch_size,                       # 无重叠
            bias=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        参数:
            pixel_values: [B, 3, H, W] 或 [B, 3, T, H, W]
        返回:
            embeddings: [B, num_patches, embed_dim]
            grid_thw: (T', H', W') — patch 网格尺寸
        """
        # ==================== 图像处理 ====================
        if pixel_values.ndim == 4:
            bsz, channels, height, width = pixel_values.shape
            if channels != 3:
                raise ValueError("pixel_values image input must have shape [B, 3, H, W].")

            # Conv2D：[B, 3, H, W] → [B, embed_dim, H', W']
            x = self.proj_image(pixel_values)

            # [B, embed_dim, H', W'] → [B, H'*W', embed_dim]
            # permute(0, 2, 3, 1)：把通道维放到最后
            x = x.permute(0, 2, 3, 1).reshape(bsz, -1, self.embed_dim)

            # H' = H / patch_size, W' = W / patch_size
            grid_thw = (1, height // self.patch_size, width // self.patch_size)
            return x, grid_thw

        # ==================== 视频处理 ====================
        if pixel_values.ndim == 5:
            bsz, channels, frames, height, width = pixel_values.shape
            if channels != 3:
                raise ValueError("pixel_values video input must have shape [B, 3, T, H, W].")
            if frames % self.temporal_patch_size != 0:
                raise ValueError("T must be divisible by temporal_patch_size in this teaching implementation.")

            # Conv3D：[B, 3, T, H, W] → [B, embed_dim, T', H', W']
            x = self.proj_video(pixel_values)

            # [B, embed_dim, T', H', W'] → [B, T'*H'*W', embed_dim]
            x = x.permute(0, 2, 3, 4, 1).reshape(bsz, -1, self.embed_dim)

            # T' = T / temporal_patch_size, H' = H / patch_size, W' = W / patch_size
            grid_thw = (
                frames // self.temporal_patch_size,
                height // self.patch_size,
                width // self.patch_size,
            )
            return x, grid_thw

        raise ValueError("pixel_values must be [B, 3, H, W] or [B, 3, T, H, W].")


def build_vision_position_embeddings(
    grid_thw: Tuple[int, int, int],
    head_dim: int,
    spatial_merge_size: int,
    rope: VisionRotaryEmbedding,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    为未合并的 patch token 构建视觉 RoPE 的 cos/sin。

    【核心设计】
    1. 位置 ID 需要匹配合并前的官方 patch 顺序
    2. 使用 rope table 在 h/w 二维网格上做 rotary embedding
    3. 时间维度通过重复 h/w 表实现（每帧复用相同的空间位置表）

    【返回】
    cos/sin: [1, 1, N, head_dim]

    【位置 ID 构建详解】
    例如 4x4 grid，spatial_merge_size=2：
    hpos_ids 应该匹配官方 patch 顺序：
    [[0,1,2,3],      原始高度位置
     [0,1,2,3],      ...（每行重复）
     [0,1,2,3],
     [0,1,2,3]]
    合并后变成 2x2：
    hpos_ids_merged:
    [[0,0,1,1],       前两个合并 patch 用高度 0
     [0,0,1,1],       ...（每行重复）
     [2,2,3,3],       前两个合并 patch 用高度 2
     [2,2,3,3]]

    这确保合并后相邻 patch 有相似的位置编码。
    """
    grid_t, grid_h, grid_w = grid_thw
    if grid_h % spatial_merge_size != 0 or grid_w % spatial_merge_size != 0:
        raise ValueError("H' and W' must be divisible by spatial_merge_size.")

    # ========== 构建 h 位置 ID ==========
    # 原始高度位置：[grid_h, grid_w]
    hpos_ids = torch.arange(grid_h, device=device).unsqueeze(1).expand(-1, grid_w)
    # 重新整形为合并后的格式：[grid_h/s, s, grid_w/s, s]
    hpos_ids = hpos_ids.reshape(
        grid_h // spatial_merge_size,
        spatial_merge_size,
        grid_w // spatial_merge_size,
        spatial_merge_size,
    )
    # 交换维度：[s, s] 变成 [s, s]，但整体变成 [grid_h/s, grid_w/s, s, s]
    hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()  # 展平成一维

    # ========== 构建 w 位置 ID（类似） ==========
    wpos_ids = torch.arange(grid_w, device=device).unsqueeze(0).expand(grid_h, -1)
    wpos_ids = wpos_ids.reshape(
        grid_h // spatial_merge_size,
        spatial_merge_size,
        grid_w // spatial_merge_size,
        spatial_merge_size,
    )
    wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()

    # ========== 组合成 [T*H*W, 2] ==========
    # 每行是 (h_id, w_id)，时间维重复
    pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(grid_t, 1)
    max_grid_size = max(grid_h, grid_w)

    # ========== 生成 RoPE 表 ==========
    # rope_table: [max_grid_size, head_dim/4]
    rope_table = rope(max_grid_size)

    # 用位置 ID 索引 rope 表：[N, 2] → [N, head_dim/2]
    # 这实现了每个 (h, w) 位置有独特的 RoPE 编码
    rotary = rope_table[pos_ids].flatten(1)  # [N, head_dim/2]

    # 复制到完整维度：[N, head_dim/2] → [N, head_dim]
    emb = torch.cat((rotary, rotary), dim=-1)

    # 健壮性处理：处理 head_dim 和 rope_table 不完全匹配的情况
    if emb.shape[-1] != head_dim:
        emb = F.pad(emb, (0, max(0, head_dim - emb.shape[-1])))[:, :head_dim]

    cos = emb.cos().to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
    sin = emb.sin().to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
    return cos, sin


def build_vision_window_mask(
    batch_size: int,
    grid_thw: Tuple[int, int, int],
    patch_size: int,
    spatial_merge_size: int,
    window_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    近似 Qwen2.5-VL 视觉窗口注意力的掩码。

    【设计思路】
    官方代码使用 window_index + cu_window_seqlens 来管理窗口注意力。
    本函数为 batch 内共享 grid 构建等效的布尔可见性掩码。

    【窗口分配策略】
    1. 每个 patch 属于一个 merged patch（s×s 合并）
    2. 每个 merged patch 属于一个窗口
       （window_size / patch_size / spatial_merge_size 个合并 patch）
    3. 同一窗口内的 patch 可以相互 attend

    【返回】
    [B, 1, N, N]，True = token 可以相互关注

    【ID 分配机制】
    用大素数构建唯一窗口 ID，避免不同维度的窗口 ID 冲突：
    window_id = t × 10_000_000 + wh × 10_000 + ww
    这样 t/wh/ww 即使值相同也不会混淆。
    """
    grid_t, grid_h, grid_w = grid_thw
    n_tokens = grid_t * grid_h * grid_w

    # 合并后的 grid 尺寸
    merged_h = grid_h // spatial_merge_size
    merged_w = grid_w // spatial_merge_size
    # 每个窗口覆盖多少个 merged patch
    merged_window = max(1, window_size // patch_size // spatial_merge_size)

    # 为每个 token 分配窗口 ID
    ids: List[int] = []
    for t in range(grid_t):
        for h in range(grid_h):
            for w in range(grid_w):
                # 找到该 patch 所属的 merged patch 位置
                mh = h // spatial_merge_size
                mw = w // spatial_merge_size
                # 找到该 merged patch 所属的窗口
                wh = mh // merged_window
                ww = mw // merged_window
                # 构建唯一窗口 ID
                window_id = (t * 10_000_000) + (wh * 10_000) + ww
                ids.append(window_id)

    window_ids = torch.tensor(ids, device=device)
    # 两个 token 可见 iff 它们的窗口 ID 相同
    visible = window_ids[:, None].eq(window_ids[None, :])
    return visible.view(1, 1, n_tokens, n_tokens).expand(batch_size, 1, n_tokens, n_tokens)


class VisionBlock(nn.Module):
    """
    Qwen2.5-VL 风格的视觉 Transformer 块。

    【结构】
    Pre-Norm + 残差连接：
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    【注意力类型】
    - Full Attention（全注意力）：全局可见，4 层（index 7, 15, 23, 31）
    - Window Attention（窗口注意力）：只可见同窗口内 token，其余层
    """

    def __init__(self, config: ModelConfig, block_idx: int):
        super().__init__()
        self.config = config
        # 判断当前层是否使用 Full Attention（全局注意力）
        self.use_full_attention = block_idx in config.full_attention_block_indices

        # Pre-Norm：LayerNorm 在注意力/MLP 之前
        self.norm1 = nn.RMSNorm(config.vision_hidden_dim, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(config.vision_hidden_dim, eps=config.rms_norm_eps)

        # 注意力：num_kv_heads == num_heads，无 GQA，is_causal=False（双向）
        self.attn = MultiHeadAttention(
            hidden_dim=config.vision_hidden_dim,
            num_heads=config.vision_num_heads,
            num_kv_heads=config.vision_num_heads,  # 无 GQA
            is_causal=False,                         # 关键：视觉注意力是双向的
            bias=True,
        )

        # SwiGLU MLP
        self.mlp = SwiGLU(
            config.vision_hidden_dim,
            config.vision_mlp_hidden_dim,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: Tuple[int, int, int],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        参数:
            x: [B, N, vision_hidden_dim] patch tokens
            grid_thw: patch 网格尺寸
            position_embeddings: (cos, sin) 视觉 RoPE
        """
        # Full Attention 层：不用窗口 mask（全局可见）
        # Window Attention 层：使用窗口 mask（只可见同窗口内 token）
        if self.use_full_attention:
            visibility = None  # 使用全 1 mask（双向）
        else:
            visibility = build_vision_window_mask(
                batch_size=x.shape[0],
                grid_thw=grid_thw,
                patch_size=self.config.patch_size,
                spatial_merge_size=self.config.spatial_merge_size,
                window_size=self.config.vision_window_size,
                device=x.device,
            )

        # Pre-Norm + Attention + 残差
        x = x + self.attn(
            self.norm1(x),
            position_embeddings=position_embeddings,
            custom_visibility_mask=visibility,
        )
        # Pre-Norm + MLP + 残差
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerger(nn.Module):
    """
    Qwen2.5-VL PatchMerger：空间 2×2 分组后接 RMSNorm + MLP。

    【双重作用】
    1. 空间压缩：每 s×s=4 个相邻 patch 合并成 1 个
       - 4× 下采样，大幅减少 token 数量
       - 224×224 → 16×16 → 8×8 = 64 tokens

    2. 维度变换：vision_hidden_dim → llm_hidden_dim
       - ViT 特征 1280 维 → LLM 3584 维
       - 跨模态对齐的关键步骤

    【与官方 HF 实现一致的要点】
    1. 先对每个 patch 的 vision_hidden_dim 维度做 RMSNorm
    2. 再把 s×s 个 patch 拼成 s²×vision_hidden_dim 维向量
    3. 通过两层 MLP 投影到 llm_hidden_dim

    【数据流示例】
    输入：[B, 256, 1280] — 16×16 patches，1280 维
    ↓ ln_q (RMSNorm)
    ↓ reshape → [B×16×16, 4, 4, 1280]
    ↓ flatten → [B×256, 5120]
    ↓ MLP → [B×256, 3584]
    输出：[B, 256, 3584]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.s = config.spatial_merge_size                    # 2
        self.context_dim = config.vision_hidden_dim          # 1280
        self.out_dim = config.llm_hidden_dim                # 3584
        self.hidden_size = self.context_dim * (self.s ** 2)  # 1280 * 4 = 5120

        # 对每个 patch 的 vision_hidden_dim 维度做 RMSNorm
        self.ln_q = nn.RMSNorm(self.context_dim, eps=config.rms_norm_eps)
        # 两层 MLP：s²×context_dim → s²×context_dim → out_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.out_dim, bias=True),
        )

    def forward(self, x: torch.Tensor, grid_thw: Tuple[int, int, int]) -> torch.Tensor:
        """
        参数:
            x: [B, T*H*W, vision_hidden_dim] 展平后的 patch tokens
            grid_thw: (T, H, W) patch 网格尺寸
        返回:
            [B, T*(H/s)*(W/s), llm_hidden_dim]
        """
        bsz, _, dim = x.shape
        grid_t, grid_h, grid_w = grid_thw
        s = self.s

        if grid_h % s != 0 or grid_w % s != 0:
            raise ValueError("grid_h and grid_w must be divisible by spatial_merge_size.")

        # ========== 1. RMSNorm ==========
        # 在 patch 合并之前对 vision_hidden_dim 维度做归一化
        x = self.ln_q(x)

        # ========== 2. Reshape 为 3D grid ==========
        # [B, T*H*W, dim] → [B, T, H, W, dim]
        x = x.view(bsz, grid_t, grid_h, grid_w, dim)

        # ========== 3. Patch 合并 ==========
        merged_h = grid_h // s  # 16
        merged_w = grid_w // s  # 16

        # [B, T, H, W, dim] → [B, T, H/s, s, W/s, s, dim]
        # 这是关键的重塑操作，把 s×s 个相邻 patch 放在相邻位置
        x = x.view(bsz, grid_t, merged_h, s, merged_w, s, dim)

        # [B, T, H/s, s, W/s, s, dim] → [B, T, H/s, W/s, s, s, dim]
        # permute(0, 1, 2, 4, 3, 5, 6) 把 s 维度移到正确位置
        x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()

        # [B, T*H/s*W/s, s*s*dim]
        x = x.view(bsz, grid_t * merged_h * merged_w, s * s * dim)

        # ========== 4. MLP 投影 ==========
        return self.mlp(x)


class VisionTransformer(nn.Module):
    """
    Qwen2.5-VL 风格的视觉塔（Vision Encoder）。

    【完整数据流】
    输入图像/视频 [B, 3, H, W]
        ↓ VisionPatchEmbed
    patch tokens [B, N, vision_hidden_dim]
        ↓ ×32 VisionBlock (with Window/Full Attention + RoPE)
    视觉特征 [B, N, vision_hidden_dim]
        ↓ PatchMerger
    压缩后特征 [B, N/s², llm_hidden_dim]

    【关键组件】
    - patch_embed：Conv2D/Conv3D 图像 → patch tokens
    - vision_rope：视觉 RoPE 位置编码
    - blocks：×32 VisionBlock（交替 Window/Full Attention）
    - merger：PatchMerger（压缩 + 维度变换）
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.patch_embed = VisionPatchEmbed(config)

        # ViT 侧的 RoPE（用于 vision encoder 内部）
        vision_head_dim = config.vision_hidden_dim // config.vision_num_heads
        self.vision_rope = VisionRotaryEmbedding(
            dim=vision_head_dim // 2,
            theta=config.vision_rope_theta,
        )

        self.blocks = nn.ModuleList(
            [VisionBlock(config, i) for i in range(config.num_vision_blocks)]
        )
        self.merger = PatchMerger(config)

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        返回:
            visual_embeds: [B, N_merged, llm_hidden_dim] 合并后的视觉 token
            grid_thw: (T', H', W') 合并后的 grid 尺寸
        """
        # 1. Patch Embedding
        x, grid_thw = self.patch_embed(pixel_values)

        # 2. 构建视觉 RoPE
        cos, sin = build_vision_position_embeddings(
            grid_thw=grid_thw,
            head_dim=self.config.vision_hidden_dim // self.config.vision_num_heads,
            spatial_merge_size=self.config.spatial_merge_size,
            rope=self.vision_rope,
            device=x.device,
            dtype=x.dtype,
        )

        # 3. 通过 Vision Blocks
        for block in self.blocks:
            x = block(x, grid_thw=grid_thw, position_embeddings=(cos, sin))

        # 4. Patch Merger：压缩 + 维度变换
        x = self.merger(x, grid_thw)
        return x, grid_thw


# ============================================================
# LLM 侧模块
# ============================================================

class DecoderLayer(nn.Module):
    """
    Qwen2.5-VL 文本解码器层。

    【结构】（与 VisionBlock 类似）
    Pre-Norm + 残差连接：
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    【与 VisionBlock 的关键区别】
    1. 使用 causal attention（只能看前面的 token）
    2. 使用 GQA（4 个 KV heads → 28 个 Q heads）
    3. 使用 MRoPE（t/h/w 三维位置编码）
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Pre-Norm
        self.input_layernorm = nn.RMSNorm(config.llm_hidden_dim, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.llm_hidden_dim, eps=config.rms_norm_eps)

        # GQA Attention：num_kv_heads=4 < num_heads=28
        self.self_attn = MultiHeadAttention(
            hidden_dim=config.llm_hidden_dim,
            num_heads=config.llm_num_attention_heads,     # 28
            num_kv_heads=config.llm_num_key_value_heads,  # 4
            is_causal=True,                              # 关键：因果注意力
            bias=True,
        )

        # SwiGLU MLP
        self.mlp = SwiGLU(config.llm_hidden_dim, config.llm_mlp_hidden_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """前向传播，完整流程：Pre-Norm → Attention → Add → Pre-Norm → MLP → Add"""
        # Attention path
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            mrope_section=self.config.mrope_section,  # 启用 MRoPE
        )
        hidden_states = residual + hidden_states

        # MLP path
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class TextDecoder(nn.Module):
    """
    Qwen2.5-VL decoder-only 语言模型主体。

    【结构】
    - embed_tokens：词表 embedding
    - layers：×28 DecoderLayer
    - norm：最终 RMSNorm
    - rotary_emb：RoPE/MRoPE 位置编码

    【前向流程】
    1. input_ids → embedding
    2. 构建 position_ids（3D：t/h/w）
    3. 生成 MRoPE cos/sin
    4. 通过 28 层 DecoderLayer
    5. RMSNorm
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.llm_hidden_dim)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_llm_blocks)])
        self.norm = nn.RMSNorm(config.llm_hidden_dim, eps=config.rms_norm_eps)
        self.rotary_emb = TextRotaryEmbedding(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        二选一：
        - input_ids：token IDs → embedding
        - inputs_embeds：直接给 embedding（用于视觉 token 注入）

        position_ids 格式：[3, B, N]
        - dim 0：temporal 位置
        - dim 1：height 位置
        - dim 2：width 位置
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        bsz, seq_len, _ = inputs_embeds.shape

        # 构建 position_ids（3D MRoPE 格式）
        if position_ids is None:
            # 基础 1D 位置：[seq_len]
            base = torch.arange(seq_len, device=inputs_embeds.device).view(1, -1)
            # [1, seq_len] → [3, B, seq_len]（三个维度相同 = 标准 1D RoPE）
            position_ids = base.expand(bsz, -1)
            position_ids = position_ids.unsqueeze(0).expand(3, bsz, seq_len)

        # 生成 MRoPE cos/sin（三个表）
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # 通过 LLM Decoder
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )
        return self.norm(hidden_states)


# ============================================================
# 完整多模态模型
# ============================================================

class Qwen25VLArchitectureSimulator(nn.Module):
    """
    Qwen2.5-VL 架构模拟器。

    【完整数据流】
    输入：
    - input_ids: 包含 <|vision_start|>, <|image_pad|>×N, <|vision_end|> 的 token IDs
    - pixel_values: 原始图像/视频张量

    处理步骤：
    1. 文本 embedding（token IDs → 向量）
    2. 视觉编码（图像 → 合并后的视觉 token）
    3. 视觉 token 注入（替换对应占位符位置）
    4. 构建 MRoPE position_ids
    5. LLM Decoder
    6. LM Head → logits

    【视觉 token 注入机制】
    这是与简单前插方案的本质区别：
    - 简单前插：把视觉 token 拼到文本序列前面
    - 真实注入：把视觉 embedding 填入 <|image_pad|> 占位符位置

    这确保视觉 token 的位置与文本 prompt 中的占位符一一对应。
    """

    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        self.visual = VisionTransformer(config)
        self.language_model = TextDecoder(config)
        self.lm_head = nn.Linear(config.llm_hidden_dim, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.language_model.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """权重初始化：正态分布，std = initializer_range"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.embed_tokens

    def _build_multimodal_position_ids(
        self,
        input_ids: torch.LongTensor,
        visual_grid_thw: Optional[Tuple[int, int, int]],
        visual_token_count: int = 0,
        is_video: bool = False,
    ) -> torch.LongTensor:
        """
        构建多模态 3D 位置 ID。

        【输出格式】
        [3, B, L] — temporal / height / width 三个维度的位置

        【文本 token】
        三个维度值相同 = 普通 1D RoPE

        【视觉占位符 token】
        三个维度用合并后网格的 t/h/w 坐标

        【位置偏移】
        视觉位置从该 token 在序列中的起始位置开始偏移，
        保证相对位置关系正确。

        【例子】
        如果 <|image_pad|> 从位置 10 开始（对应 4 个视觉 token），
        这些位置的 MRoPE 位置 ID 不是简单的 10, 11, 12, 13，
        而是 (0,0,0), (0,0,1), (0,1,0), (0,1,1)（2×2 网格的坐标）
        加上起始偏移 10。
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # 初始化为 1D 位置（三个维度相同）
        base = torch.arange(seq_len, device=device).view(1, seq_len).expand(bsz, seq_len)
        position_ids = base.unsqueeze(0).expand(3, bsz, seq_len).clone()

        if visual_grid_thw is None or visual_token_count == 0:
            return position_ids

        # 验证 visual_token_count 与合并后网格一致
        grid_t, grid_h, grid_w = visual_grid_thw
        s = self.config.spatial_merge_size
        llm_grid_h = grid_h // s
        llm_grid_w = grid_w // s
        expected = grid_t * llm_grid_h * llm_grid_w
        if expected != visual_token_count:
            raise ValueError(
                f"visual_token_count={visual_token_count}, but merged grid gives {expected}."
            )

        # 找到视觉占位符位置
        token_id = self.config.video_token_id if is_video else self.config.image_token_id
        visual_mask = input_ids.eq(token_id)

        # 构建视觉位置的 3D 坐标
        coords = []
        for t in range(grid_t):
            for h in range(llm_grid_h):
                for w in range(llm_grid_w):
                    coords.append((t, h, w))
        coords = torch.tensor(coords, device=device, dtype=torch.long)

        # 为每个 batch 填充视觉位置的三维坐标
        for b in range(bsz):
            idx = visual_mask[b].nonzero(as_tuple=False).flatten()
            if idx.numel() != expected:
                raise ValueError(
                    f"Batch {b}: found {idx.numel()} visual placeholder tokens, "
                    f"but visual encoder produced {expected} merged tokens."
                )
            start = idx[0]
            # 用序列起始位置偏移三个轴的坐标
            position_ids[0, b, idx] = start + coords[:, 0]  # temporal
            position_ids[1, b, idx] = start + coords[:, 1]  # height
            position_ids[2, b, idx] = start + coords[:, 2]  # width

        return position_ids

    def _inject_visual_embeddings(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        visual_embeds: torch.Tensor,
        is_video: bool,
    ) -> torch.Tensor:
        """
        将视觉 embedding 注入到文本 embedding 中。

        【核心设计】
        不是简单地把视觉 token 拼到文本前面，
        而是用视觉 encoder 输出替换 <|image_pad|> 占位符位置。

        【验证】
        确保 <|image_pad|> 占位符数量 == 合并后视觉 token 数量。
        每个占位符位置被对应序号的视觉 token 替换。

        【为什么这样做】
        1. 保持序列长度不变
        2. 视觉 token 自然融入文本序列
        3. 与真实 Qwen2.5-VL 行为一致
        """
        token_id = self.config.video_token_id if is_video else self.config.image_token_id
        visual_mask = input_ids.eq(token_id)

        bsz, visual_len, hidden = visual_embeds.shape
        if inputs_embeds.shape[-1] != hidden:
            raise ValueError("visual_embeds hidden size must match LLM hidden size.")

        out = inputs_embeds.clone()
        for b in range(bsz):
            idx = visual_mask[b].nonzero(as_tuple=False).flatten()
            if idx.numel() != visual_len:
                raise ValueError(
                    f"Batch {b}: placeholder count {idx.numel()} != visual token count {visual_len}. "
                    "Create input_ids with exactly one image/video placeholder per merged visual token."
                )
            out[b, idx, :] = visual_embeds[b].to(dtype=out.dtype)
        return out

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        is_video: bool = False,
    ) -> torch.Tensor:
        """
        前向传播。

        参数:
            input_ids: [B, L] — 包含视觉占位符的 token IDs
            pixel_values: [B, 3, H, W] 或 [B, 3, T, H, W] — 图像/视频
            attention_mask: [B, L] — attention mask
            is_video: 是否是视频模式

        返回:
            logits: [B, L, vocab_size]
        """
        # 1. 文本 embedding
        inputs_embeds = self.language_model.embed_tokens(input_ids)

        visual_grid_thw = None
        visual_token_count = 0

        # 2. 视觉编码 + 注入
        if pixel_values is not None:
            visual_embeds, visual_grid_thw = self.visual(pixel_values)
            visual_token_count = visual_embeds.shape[1]
            # 用视觉 embedding 替换 <|image_pad|> 占位符位置
            inputs_embeds = self._inject_visual_embeddings(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                visual_embeds=visual_embeds,
                is_video=is_video,
            )

        # 3. 构建 MRoPE position_ids
        position_ids = self._build_multimodal_position_ids(
            input_ids=input_ids,
            visual_grid_thw=visual_grid_thw,
            visual_token_count=visual_token_count,
            is_video=is_video,
        )

        # 4. LLM Decoder
        hidden_states = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        # 5. LM Head
        return self.lm_head(hidden_states)

    @torch.no_grad()
    def generate_greedy(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        is_video: bool = False,
    ) -> torch.LongTensor:
        """
        简化的贪婪生成（演示用）。

        【警告】
        这是一个非常基础的实现，每步都重新计算视觉编码，无 KV cache。
        仅用于教学演示，生产级实现需要：
        - KV Cache
        - 批处理优化
        - 多种采样策略
        - Stop tokens 检测
        - 等等...

        【贪婪生成】
        每步取概率最高的 token：
            next_token = argmax P(token | previous_tokens)
        """
        self.eval()
        generated = input_ids
        for _ in range(max_new_tokens):
            logits = self(
                input_ids=generated,
                pixel_values=pixel_values,
                is_video=is_video,
            )
            # 取概率最高的 token
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            # 检测是否全部生成结束
            if (next_token == self.config.eos_token_id).all():
                break
        return generated


# ============================================================
# 演示辅助函数
# ============================================================

def make_demo_config() -> ModelConfig:
    """
    创建小型配置：保留架构比例但大幅减少参数量。

    官方 7B → demo 版本约 20M 参数，可在普通硬件上快速运行。

    【主要缩减】
    - vision_hidden_dim: 1280 → 128 (10×)
    - llm_hidden_dim: 3584 → 256 (14×)
    - num_vision_blocks: 32 → 4 (8×)
    - num_llm_blocks: 28 → 4 (7×)
    """
    return ModelConfig(
        # vision
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        vision_hidden_dim=128,           # 官方 1280 → 128
        vision_mlp_hidden_dim=256,        # 官方 3420 → 256
        num_vision_blocks=4,              # 官方 32 → 4
        vision_num_heads=4,               # 官方 16 → 4
        vision_window_size=56,             # 官方 112 → 56
        full_attention_block_indices=(1, 3),
        # text
        llm_hidden_dim=256,               # 官方 3584 → 256
        llm_mlp_hidden_dim=512,           # 官方 18944 → 512
        num_llm_blocks=4,                # 官方 28 → 4
        llm_num_attention_heads=8,        # 官方 28 → 8
        llm_num_key_value_heads=2,       # 官方 4 → 2
        vocab_size=152064,
        max_position_embeddings=4096,
        # RoPE
        rope_theta=1_000_000.0,
        vision_rope_theta=10_000.0,
        # head_dim = 256 / 8 = 32；真实 (16,24,24)*2 需要 head_dim=128
        # 小配置下会 fallback 到平均分段
        mrope_section=(16, 24, 24),
    )


def build_demo_input_ids(
    config: ModelConfig,
    batch_size: int,
    visual_token_count: int,
    text_len_after_image: int = 8,
    is_video: bool = False,
    device: Optional[torch.device] = None,
) -> torch.LongTensor:
    """
    创建带有视觉占位符的 prompt。

    【Token 布局】
    [BOS] [vision_start] [image_pad]×N [vision_end] [text tokens...]

    其中 N = visual_token_count（合并后的视觉 token 数量）
    <|image_pad|> 用 image_token_id 表示。

    【例子】
    56×56 图像：
    - patch grid: 4×4 = 16
    - merged grid: 2×2 = 4 tokens
    - 序列长度：1(BOS) + 1(vision_start) + 4(image_pad) + 1(vision_end) + 8(text) = 15
    """
    device = device or torch.device("cpu")
    visual_token_id = config.video_token_id if is_video else config.image_token_id

    # BOS + vision_start
    prefix = torch.tensor(
        [config.bos_token_id, config.vision_start_token_id],
        dtype=torch.long,
        device=device,
    )
    # N 个 image_pad 占位符
    placeholders = torch.full(
        (visual_token_count,),
        visual_token_id,
        dtype=torch.long,
        device=device,
    )
    # vision_end
    suffix = torch.tensor([config.vision_end_token_id], dtype=torch.long, device=device)
    # 随机文本 token
    text = torch.randint(
        low=0,
        high=min(10_000, config.vocab_size),
        size=(text_len_after_image,),
        dtype=torch.long,
        device=device,
    )

    one = torch.cat([prefix, placeholders, suffix, text], dim=0)
    return one.unsqueeze(0).expand(batch_size, -1).contiguous()


def demo() -> None:
    """演示模型前向传播"""
    torch.manual_seed(0)
    config = make_demo_config()
    model = Qwen25VLArchitectureSimulator(config)

    # 56×56 图像 → patch grid 4×4 → merged grid 2×2 → 4 visual tokens
    image = torch.randn(1, 3, 56, 56)

    # 只运行视觉编码器看看输出
    with torch.no_grad():
        visual_embeds, grid_thw = model.visual(image)

    print("=" * 50)
    print("Qwen2.5-VL 架构模拟器演示")
    print("=" * 50)
    print(f"\n输入图像: {tuple(image.shape)}")
    print(f"grid_thw (patch网格): {grid_thw}")
    print(f"合并后视觉 embedding: {tuple(visual_embeds.shape)}")

    # 构建带视觉占位符的 input_ids
    input_ids = build_demo_input_ids(
        config=config,
        batch_size=1,
        visual_token_count=visual_embeds.shape[1],
        text_len_after_image=6,
        is_video=False,
        device=image.device,
    )

    # 完整前向传播
    logits = model(input_ids=input_ids, pixel_values=image)

    print(f"\ninput_ids shape: {tuple(input_ids.shape)}")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"  = [batch={logits.shape[0]}, seq_len={logits.shape[1]}, vocab={logits.shape[2]}]")
    print("\n演示完成!")


if __name__ == "__main__":
    demo()
