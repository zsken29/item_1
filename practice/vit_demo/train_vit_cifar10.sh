#!/bin/bash
# ViT CIFAR-10 训练脚本
# 使用方法: cd C:/codes/python/items_1/practice/vit_demo && python train_vit_cifar10.py

echo "=========================================="
echo "ViT CIFAR-10 完整训练"
echo "=========================================="
echo ""
echo "配置:"
echo "  - 图像尺寸: 32x32"
echo "  - Patch 大小: 4x4 (64 patches)"
echo "  - Embed dim: 256"
echo "  - Transformer 层数: 4"
echo "  - 训练轮数: 30"
echo "  - 批量大小: 128"
echo ""
echo "数据集: CIFAR-10 (60,000 张图片)"
echo "=========================================="

cd "C:/codes/python/items_1/practice/vit_demo"

python train_cifar10.py
