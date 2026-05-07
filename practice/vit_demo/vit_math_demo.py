"""
ViT 完整手算验证脚本 v2
使用 in_channels=3，固定种子，固定权重，每步打印中间结果
"""

import torch
import torch.nn as nn
import math

# ============================================================
# 第1步：设置全局固定种子，保证完全可复现
# ============================================================
torch.manual_seed(123)

# ============================================================
# 第2步：定义简化的 ViT（与 vit_model.py 结构一致，但添加打印）
# ============================================================

class PatchEmbedding(nn.Module):
    """将图像分割成 patches 并进行线性投影"""

    def __init__(self, img_size=4, patch_size=2, in_channels=3, embed_dim=6):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        print(f"\n{'='*60}")
        print(f"[Patch Embedding]")
        print(f"  输入形状: {x.shape}  (B, C, H, W)")
        print(f"  输入值:\n{x}")

        x = self.proj(x)
        print(f"  Conv2d 输出形状: {x.shape}  (B, embed_dim, H/p, W/p)")

        x = x.flatten(2)
        print(f"  flatten(2) 后形状: {x.shape}  (B, embed_dim, n_patches)")

        x = x.transpose(1, 2)
        print(f"  transpose(1,2) 后形状: {x.shape}  (B, n_patches, embed_dim)")
        print(f"  Patch embeddings:\n{x}")

        return x


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""

    def __init__(self, embed_dim=6, num_heads=3, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        print(f"\n{'='*60}")
        print(f"[Multi-Head Attention]")
        print(f"  输入形状: {x.shape}  (B, N, C)")
        B, N, C = x.shape

        qkv = self.qkv(x)
        print(f"  QKV 投影后形状: {qkv.shape}  (B, N, C*3)")
        print(f"  QKV 值:\n{qkv}")

        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        print(f"  Reshaped QKV 形状: {qkv.shape}  (3, B, num_heads, N, head_dim)")

        q, k, v = qkv[0], qkv[1], qkv[2]
        print(f"  Q 形状: {q.shape}, K 形状: {k.shape}, V 形状: {v.shape}")
        print(f"  Q 值:\n{q}")
        print(f"  K 值:\n{k}")
        print(f"  V 值:\n{v}")

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        print(f"  注意力分数形状: {attn.shape}  (B, num_heads, N, N)")
        print(f"  注意力分数:\n{attn}")

        attn = attn.softmax(dim=-1)
        print(f"  Softmax 后:\n{attn}")

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        print(f"  加权求和后形状: {x.shape}")

        x = self.proj(x)
        print(f"  投影后形状: {x.shape}")
        print(f"  注意力输出:\n{x}")

        return x


class MLP(nn.Module):
    """MLP 块 (FFN)"""

    def __init__(self, embed_dim=6, hidden_dim=12, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        print(f"\n{'='*60}")
        print(f"[MLP]")
        print(f"  输入形状: {x.shape}")

        x = self.fc1(x)
        print(f"  FC1 输出形状: {x.shape}")
        print(f"  FC1 输出:\n{x}")

        x = self.act(x)
        print(f"  GELU 后:\n{x}")

        x = self.fc2(x)
        print(f"  FC2 输出形状: {x.shape}")
        print(f"  FC2 输出:\n{x}")

        return x


class TransformerBlock(nn.Module):
    """Transformer 编码器块"""

    def __init__(self, embed_dim=6, num_heads=3, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)

    def forward(self, x):
        print(f"\n{'='*60}")
        print(f"[Transformer Block]")

        residual = x
        x = self.norm1(x)
        print(f"  norm1 后形状: {x.shape}")
        x = self.attn(x)
        x = residual + x
        print(f"  注意力残差连接后形状: {x.shape}")

        residual = x
        x = self.norm2(x)
        print(f"  norm2 后形状: {x.shape}")
        x = self.mlp(x)
        x = residual + x
        print(f"  MLP 残差连接后形状: {x.shape}")

        return x


class SimpleViT(nn.Module):
    """简化版 ViT 模型（用于手算验证）"""

    def __init__(self, img_size=4, patch_size=2, in_channels=3, num_classes=2,
                 embed_dim=6, depth=1, num_heads=3, mlp_ratio=2.0, dropout=0.0):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        print(f"\n{'='*60}")
        print(f"[Vision Transformer 完整流程]")

        B = x.shape[0]
        print(f"  原始输入: {x.shape}")

        x = self.patch_embed(x)
        print(f"  Patch embedding 输出: {x.shape}")

        cls_tokens = self.cls_token.expand(B, -1, -1)
        print(f"  CLS token: {cls_tokens.shape}")
        x = torch.cat([cls_tokens, x], dim=1)
        print(f"  添加 CLS token 后: {x.shape}")
        print(f"  带 CLS token:\n{x}")

        x = x + self.pos_embed
        print(f"  添加位置编码后: {x.shape}")
        print(f"  带位置编码:\n{x}")
        x = self.pos_drop(x)

        for i, block in enumerate(self.blocks):
            print(f"\n--- Transformer Block {i+1} ---")
            x = block(x)

        x = self.norm(x)
        print(f"\n{'='*60}")
        print(f"[最终处理]")
        print(f"  norm 后形状: {x.shape}")

        cls_token_final = x[:, 0]
        print(f"  取 [CLS] token: {cls_token_final.shape}")
        print(f"  CLS token 值: {cls_token_final}")

        logits = self.head(cls_token_final)
        print(f"  分类 logits: {logits.shape}")
        print(f"  Logits 值: {logits}")

        return logits


# ============================================================
# 第3步：设置固定权重（关键！确保手算与代码一致）
# ============================================================

def set_deterministic_weights(model):
    """设置确定性权重，方便手算验证"""
    with torch.no_grad():
        # ---------- Patch Embedding: 权重全 1，偏置全 0 ----------
        nn.init.constant_(model.patch_embed.proj.weight, 1.0)
        nn.init.zeros_(model.patch_embed.proj.bias)

        # ---------- CLS token: 固定值 ----------
        # 形状 [1, 1, 6]
        with torch.no_grad():
            model.cls_token.copy_(torch.tensor([[[1., 2., 3., 4., 5., 6.]]]))

        # ---------- 位置编码: 简单线性递增 ----------
        # 形状 [1, 5, 6]
        pos_embed_data = torch.zeros(1, 5, 6)
        for i in range(5):  # 5 个位置 (1 CLS + 4 patches)
            for j in range(6):  # 6 维 embed_dim
                pos_embed_data[0, i, j] = (i + 1) * (j + 1)
        with torch.no_grad():
            model.pos_embed.copy_(pos_embed_data)

        # ---------- QKV 投影 ----------
        # QKV 线性层: Linear(6, 18)，权重形状 [18, 6]
        # 要让 Q = x, K = x, V = x，需要设置权重使得 x @ W^T = x
        # 即 W^T = I，所以 W = I，即权重是 [6, 6] 的单位矩阵重复 3 次
        # 实际上 PyTorch 存储为 [18, 6]，我们设置 [18, 6] 的块对角单位矩阵
        qkv_weight = torch.zeros(18, 6)
        for i in range(6):
            qkv_weight[i, i] = 1.0       # Q 的权重（6行6列单位阵）
            qkv_weight[i+6, i] = 1.0     # K 的权重（6行6列单位阵）
            qkv_weight[i+12, i] = 1.0    # V 的权重（6行6列单位阵）
        with torch.no_grad():
            model.blocks[0].attn.qkv.weight.copy_(qkv_weight)
            model.blocks[0].attn.qkv.bias.zero_()
        print(f"\n[权重设置] QKV 权重形状: {model.blocks[0].attn.qkv.weight.shape}")
        print(f"QKV 权重（前2行）:\n{model.blocks[0].attn.qkv.weight[:2]}")

        # ---------- Attention 输出投影 ----------
        # 单位矩阵，让输出 = 输入
        nn.init.eye_(model.blocks[0].attn.proj.weight)
        model.blocks[0].attn.proj.bias.zero_()

        # ---------- MLP ----------
        # 单位矩阵
        nn.init.eye_(model.blocks[0].mlp.fc1.weight)
        model.blocks[0].mlp.fc1.bias.zero_()
        nn.init.eye_(model.blocks[0].mlp.fc2.weight)
        model.blocks[0].mlp.fc2.bias.zero_()

        # ---------- LayerNorm ----------
        model.blocks[0].norm1.weight.data.fill_(1.0)
        model.blocks[0].norm1.bias.data.zero_()
        model.blocks[0].norm2.weight.data.fill_(1.0)
        model.blocks[0].norm2.bias.data.zero_()

        # 最终 LayerNorm
        model.norm.weight.data.fill_(1.0)
        model.norm.bias.data.zero_()

        # ---------- 分类头 ----------
        nn.init.eye_(model.head.weight)
        model.head.bias.zero_()


# ============================================================
# 第4步：创建输入数据（明确排布）
# ============================================================
print("\n" + "="*60)
print("创建输入数据")
print("="*60)

# 使用明确排布的数据，确保我们知道每个位置的数值
# 创建 [1, 3, 4, 4] 的数据，按 C, H, W 排布
x = torch.arange(1, 49).reshape(1, 3, 4, 4).float()

print(f"输入形状: {x.shape}")
print(f"\n通道 0 (Channel 0):\n{x[0, 0]}")
print(f"\n通道 1 (Channel 1):\n{x[0, 1]}")
print(f"\n通道 2 (Channel 2):\n{x[0, 2]}")
print(f"\n完整张量:\n{x}")

# ============================================================
# 第5步：创建模型并设置权重
# ============================================================
model = SimpleViT(
    img_size=4,
    patch_size=2,
    in_channels=3,
    num_classes=2,
    embed_dim=6,
    depth=1,
    num_heads=3,
    mlp_ratio=2.0,
    dropout=0.0
)

set_deterministic_weights(model)

# ============================================================
# 第6步：运行模型
# ============================================================
output = model(x)

# ============================================================
# 第7步：打印参数量
# ============================================================
print(f"\n{'='*60}")
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"最终输出形状: {output.shape}")
print("="*60)
