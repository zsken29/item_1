"""
ViT (Vision Transformer) 演示脚本
演示如何使用 Vision Transformer 进行图像分类
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

from vit_model import (
    VisionTransformer,
    vit_tiny, vit_small, vit_base,
    PatchEmbedding, MultiHeadAttention
)


def demo_patch_embedding():
    """演示图像如何被分割成 patches"""
    print("=" * 50)
    print("1. Patch Embedding 演示")
    print("=" * 50)

    patch_embed = PatchEmbedding(img_size=224, patch_size=16, embed_dim=768)
    print(f"图像尺寸: 224x224")
    print(f"Patch 大小: 16x16")
    print(f"Patch 数量: {patch_embed.n_patches}")
    print(f"Embedding 维度: 768")

    # 模拟一张图像
    x = torch.randn(1, 3, 224, 224)
    patches = patch_embed(x)
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状 (n_patches, embed_dim): {patches.shape}")


def demo_attention():
    """演示多头注意力机制"""
    print("\n" + "=" * 50)
    print("2. 多头自注意力演示")
    print("=" * 50)

    B, N, C = 2, 16, 768  # Batch, Sequence Length, Embed Dim
    num_heads = 12

    attn = MultiHeadAttention(embed_dim=C, num_heads=num_heads)
    x = torch.randn(B, N, C)

    with torch.no_grad():
        # 只看 attention weights
        qkv = attn.qkv(x).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_weights = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(C // num_heads), dim=-1)

    print(f"输入形状: [B, N, C] = {x.shape}")
    print(f"Attention 权重形状: {attn_weights.shape}")
    print(f"Attention 头数: {num_heads}")

    # 可视化第一个样本的第一个头的 attention
    plt.figure(figsize=(8, 6))
    plt.imshow(attn_weights[0, 0].numpy(), cmap='viridis')
    plt.title('Attention Weights (Head 0, Sample 0)')
    plt.colorbar()
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.savefig('attention_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Attention 热力图已保存到 attention_demo.png")


def demo_model_variants():
    """演示不同规模的 ViT 模型"""
    print("\n" + "=" * 50)
    print("3. ViT 模型变体对比")
    print("=" * 50)

    variants = [
        ("ViT-Tiny", vit_tiny(num_classes=1000)),
        ("ViT-Small", vit_small(num_classes=1000)),
        ("ViT-Base", vit_base(num_classes=1000)),
    ]

    print(f"{'模型':<12} {'参数量':>15} {'ImageNet Top-1':>15}")
    print("-" * 45)
    print(f"{'ViT-Tiny':<12} {'5.7M':>15} {'71.6%':>15}")
    print(f"{'ViT-Small':<12} {'22.1M':>15} {'80.6%':>15}")
    print(f"{'ViT-Base':<12} {'86.4M':>15} {'84.5%':>15}")

    for name, model in variants:
        params = sum(p.numel() for p in model.parameters())
        print(f"\n{name}: {params:,} 参数")


def demo_forward_pass():
    """演示完整的前向传播"""
    print("\n" + "=" * 50)
    print("4. 前向传播演示")
    print("=" * 50)

    model = vit_tiny(num_classes=10)
    x = torch.randn(4, 3, 224, 224)

    model.eval()
    with torch.no_grad():
        output = model(x)

    print(f"输入批次: {x.shape}")
    print(f"模型输出: {output.shape}")
    print(f"预测类别: {output.argmax(dim=-1).tolist()}")


def demo_patch_visualization():
    """可视化图像如何被分割成 patches"""
    print("\n" + "=" * 50)
    print("5. Patch 可视化")
    print("=" * 50)

    from PIL import Image

    # 创建示例图像
    img = Image.new('RGB', (224, 224), color=(100, 150, 200))
    img_array = np.array(img)

    patch_size = 16
    n_patches_h = 224 // patch_size
    n_patches_w = 224 // patch_size

    # 绘制 grid
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img_array)

    for i in range(n_patches_h):
        for j in range(n_patches_w):
            rect = plt.Rectangle(
                [j * patch_size, i * patch_size],
                patch_size, patch_size,
                fill=False, edgecolor='red', linewidth=1
            )
            ax.add_patch(rect)

    ax.set_title(f'Image Patches ({n_patches_h}x{n_patches_w} = 196 patches)')
    ax.axis('off')
    plt.savefig('patches_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Patch 可视化已保存到 patches_demo.png")


if __name__ == "__main__":
    import math

    demo_patch_embedding()
    demo_attention()
    demo_model_variants()
    demo_forward_pass()
    demo_patch_visualization()

    print("\n" + "=" * 50)
    print("演示完成！")
    print("=" * 50)
