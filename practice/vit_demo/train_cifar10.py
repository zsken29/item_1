"""
ViT on CIFAR-10: Complete Training & Inference
Features: Progress bar, mixed precision, gradient clipping, optimized data loading
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
from tqdm import tqdm

from vit_model import VisionTransformer


# ============================================================================
# Configuration
# ============================================================================
class Config:
    data_dir = './data'
    img_size = 32
    num_classes = 10

    # ViT for CIFAR-10
    patch_size = 4
    embed_dim = 256
    depth = 4
    num_heads = 4
    mlp_ratio = 4
    dropout = 0.1

    # Training
    batch_size = 128
    epochs = 30
    lr = 3e-4
    weight_decay = 0.01
    warmup_epochs = 5
    grad_clip = 1.0

    # Optimizations
    use_amp = True
    num_workers = 4

    seed = 42
    save_dir = './checkpoints'


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
# Data Loading
# ============================================================================
def get_data_loaders(cfg):
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
        shuffle=True, num_workers=cfg.num_workers,
        pin_memory=True, persistent_workers=True
    )

    testloader = DataLoader(
        testset, batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers,
        pin_memory=True, persistent_workers=True
    )

    return trainloader, testloader


def create_vit_cifar10(cfg):
    return VisionTransformer(
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


# ============================================================================
# Learning Rate Scheduler (Warmup + Cosine Decay)
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
            factor = (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + np.cos(np.pi * progress))

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = max(self.min_lr, base_lr * factor)


# ============================================================================
# Training Functions
# ============================================================================
def train_epoch(model, loader, criterion, optimizer, device, scaler, grad_clip):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})

    return total_loss / len(loader), 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Evaluating", leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})

    return total_loss / len(loader), 100. * correct / total


# ============================================================================
# Training Loop
# ============================================================================
def train(cfg):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Mixed Precision: {cfg.use_amp and torch.cuda.is_available()}")

    print("[INFO] Loading CIFAR-10...")
    trainloader, testloader = get_data_loaders(cfg)

    model = create_vit_cifar10(cfg).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model params: {params:,}")
    print(f"[INFO] Patch: {cfg.patch_size}x{cfg.patch_size}, Patches: {(cfg.img_size // cfg.patch_size) ** 2}")
    print(f"[INFO] Embed: {cfg.embed_dim}, Depth: {cfg.depth}, Heads: {cfg.num_heads}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, cfg.warmup_epochs, cfg.epochs)

    scaler = torch.amp.GradScaler('cuda') if cfg.use_amp and torch.cuda.is_available() else None

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'epochs': []}

    best_acc = 0
    start_time = time.time()

    print("\n" + "=" * 70)
    print(f"{'Epoch':>6} | {'LR':>10} | {'Train Loss':>10} | {'Train Acc':>8} | {'Test Loss':>10} | {'Test Acc':>8} | {'Best':>8}")
    print("-" * 70)

    for epoch in range(cfg.epochs):
        epoch_start = time.time()
        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, device, scaler, cfg.grad_clip)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)

        history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, os.path.join(cfg.save_dir, 'best_vit_cifar10.pth'))

        epoch_time = time.time() - epoch_start
        print(f"{epoch+1:>6} | {current_lr:>10.2e} | {train_loss:>10.4f} | {train_acc:>7.2f}% | {test_loss:>10.4f} | {test_acc:>7.2f}% | {best_acc:>7.2f}% | {epoch_time:>5.1f}s")

    total_time = time.time() - start_time
    print("-" * 70)
    print(f"Training completed in {total_time/60:.1f} minutes | Best Test Accuracy: {best_acc:.2f}%")
    print("=" * 70)

    plot_training_history(history)

    return model, history


# ============================================================================
# Visualization
# ============================================================================
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = history['epochs']

    ax1.plot(epochs, history['train_loss'], label='Train', linewidth=2)
    ax1.plot(epochs, history['test_loss'], label='Test', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], label='Train', linewidth=2)
    ax2.plot(epochs, history['test_acc'], label='Test', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[INFO] Training curve saved to training_history.png")


# ============================================================================
# Inference Demo
# ============================================================================
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


@torch.no_grad()
def inference_demo(cfg, num_samples=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_vit_cifar10(cfg).to(device)
    checkpoint = torch.load(os.path.join(cfg.save_dir, 'best_vit_cifar10.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    _, testloader = get_data_loaders(cfg)

    images, labels = next(iter(testloader))
    images, labels = images[:num_samples].to(device), labels[:num_samples]

    with torch.amp.autocast(device_type='cuda'):
        outputs = model(images)
    _, predicted = outputs.max(1)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        true_label = CIFAR10_CLASSES[labels[i]]
        pred_label = CIFAR10_CLASSES[predicted[i]]
        color = 'green' if true_label == pred_label else 'red'

        ax.imshow(img)
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('inference_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 50)
    print("Inference Results:")
    print("=" * 50)
    correct = 0
    for i in range(num_samples):
        true = CIFAR10_CLASSES[labels[i]]
        pred = CIFAR10_CLASSES[predicted[i]]
        status = "OK" if true == pred else "NG"
        if true == pred:
            correct += 1
        print(f"  Sample {i+1:2d}: True={true:>10s} | Pred={pred:>10s} | {status}")
    print("=" * 50)
    print(f"  Accuracy: {correct}/{num_samples} ({100.*correct/num_samples:.1f}%)")
    print("[INFO] Inference demo saved to inference_demo.png")


def main():
    cfg = Config()

    model, history = train(cfg)
    inference_demo(cfg)

    print("\n" + "=" * 50)
    print("ViT CIFAR-10 Training Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
