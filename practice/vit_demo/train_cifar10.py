"""
ViT 在 CIFAR-10 上的完整训练和推理
完整流程：数据加载 → 模型训练 → 模型评估 → 推理演示
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from vit_model import VisionTransformer


# ============================================================================
# 配置
# ============================================================================
class Config:
    # 数据集配置
    data_dir = './data'
    img_size = 32          # CIFAR-10 图像尺寸
    num_classes = 10

    # ViT 配置 (适配 CIFAR-10)
    patch_size = 4         # 32/4 = 8 patches 每边
    embed_dim = 256        # 适中规模
    depth = 4              # 4 层 (轻量)
    num_heads = 4
    mlp_ratio = 4
    dropout = 0.1

    # 训练配置
    batch_size = 128
    epochs = 30
    lr = 3e-4
    weight_decay = 0.01
    warmup_epochs = 5

    # 其他
    seed = 42
    save_dir = './checkpoints'


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
# 数据加载
# ============================================================================
def get_data_loaders(cfg):
    """获取 CIFAR-10 数据加载器"""

    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
    ])

    # 下载并加载数据集
    trainset = torchvision.datasets.CIFAR10(
        root=cfg.data_dir, train=True,
        download=True, transform=train_transform
    )

    testset = torchvision.datasets.CIFAR10(
        root=cfg.data_dir, train=False,
        download=True, transform=test_transform
    )

    trainloader = DataLoader(
        trainset, batch_size=cfg.batch_size,
        shuffle=True, num_workers=2, pin_memory=True
    )

    testloader = DataLoader(
        testset, batch_size=cfg.batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )

    return trainloader, testloader


# ============================================================================
# 模型
# ============================================================================
def create_vit_cifar10(cfg):
    """创建适配 CIFAR-10 的 ViT 模型"""
    model = VisionTransformer(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        in_channels=3,
        num_classes=cfg.num_classes,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout
    )
    return model


# ============================================================================
# 学习率调度 (Linear Warmup + Cosine Decay)
# ============================================================================
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            factor = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + np.cos(np.pi * progress))

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = max(self.min_lr, base_lr * factor)


# ============================================================================
# 训练
# ============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return total_loss / len(loader), 100. * correct / total


def train(cfg):
    """完整训练流程"""
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据
    print("加载 CIFAR-10 数据集...")
    trainloader, testloader = get_data_loaders(cfg)

    # 模型
    model = create_vit_cifar10(cfg).to(device)
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Patch size: {cfg.patch_size}x{cfg.patch_size}")
    print(f"  - Patch 数量: {(cfg.img_size // cfg.patch_size) ** 2}")
    print(f"  - Embed dim: {cfg.embed_dim}")

    # 损失和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    scheduler = WarmupCosineScheduler(optimizer, cfg.warmup_epochs, cfg.epochs)

    # 训练记录
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'epochs': []
    }

    best_acc = 0
    start_time = time.time()

    print("\n开始训练...")
    print("-" * 70)

    for epoch in range(cfg.epochs):
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)

        history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, os.path.join(cfg.save_dir, 'best_vit_cifar10.pth'))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{cfg.epochs} | "
                  f"LR: {current_lr:.2e} | "
                  f"Train: {train_acc:.2f}% | "
                  f"Test: {test_acc:.2f}% | "
                  f"Best: {best_acc:.2f}%")

    total_time = time.time() - start_time
    print("-" * 70)
    print(f"训练完成! 耗时: {total_time/60:.1f} 分钟")
    print(f"最佳测试准确率: {best_acc:.2f}%")

    # 保存训练曲线
    plot_training_history(history)

    return model, history


# ============================================================================
# 可视化
# ============================================================================
def plot_training_history(history):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = history['epochs']

    # Loss 曲线
    ax1.plot(epochs, history['train_loss'], label='Train')
    ax1.plot(epochs, history['test_loss'], label='Test')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy 曲线
    ax2.plot(epochs, history['train_acc'], label='Train')
    ax2.plot(epochs, history['test_acc'], label='Test')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("训练曲线已保存到 training_history.png")


# ============================================================================
# 推理演示
# ============================================================================
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def inference_demo(cfg, num_samples=16):
    """推理演示"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载最佳模型
    model = create_vit_cifar10(cfg).to(device)
    checkpoint = torch.load(os.path.join(cfg.save_dir, 'best_vit_cifar10.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 加载测试数据
    _, testloader = get_data_loaders(cfg)

    # 获取一批样本
    images, labels = next(iter(testloader))
    images, labels = images[:num_samples].to(device), labels[:num_samples]

    # 推理
    with torch.no_grad():
        outputs = model(images)
        _, predicted = outputs.max(1)

    # 可视化
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())  # 反标准化

        true_label = CIFAR10_CLASSES[labels[i]]
        pred_label = CIFAR10_CLASSES[predicted[i]]
        color = 'green' if true_label == pred_label else 'red'

        ax.imshow(img)
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('inference_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n推理结果已保存到 inference_demo.png")

    # 打印预测详情
    print("\n" + "=" * 50)
    print("推理结果:")
    print("=" * 50)
    for i in range(num_samples):
        true = CIFAR10_CLASSES[labels[i]]
        pred = CIFAR10_CLASSES[predicted[i]]
        status = "✓" if true == pred else "✗"
        print(f"样本 {i+1:2d}: 真实={true:>10s} | 预测={pred:>10s} | {status}")


def main():
    cfg = Config()

    # 训练
    model, history = train(cfg)

    # 推理演示
    inference_demo(cfg)

    print("\n" + "=" * 50)
    print("ViT CIFAR-10 训练演示完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
