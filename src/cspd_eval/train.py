"""Evaluation: train a classifier on a distilled dataset and evaluate on the real test set.

Protocol is ported from the MGD³ reference eval
(https://github.com/jachansantiago/mode_guidance):
- SGD optimizer, lr=0.01, momentum=0.9, weight_decay=5e-4
- MultiStepLR at 2/3 and 5/6 of total epochs, gamma=0.2
- CutMix augmentation (beta=1.0, mix_p=1.0)
- ImageNet augmentation: RandomResizedCrop + HorizontalFlip + ColorJitter + PCA Lighting
- Epochs determined by IPC: IPC<=10 → 2000, IPC<=50 → 1500, etc.
- Three architectures: ConvNet-6, ResNet-18, ResNetAP-10
- Report best top-1 accuracy on real validation set

(The upstream repo is named "mode_guidance" but this is only the eval
protocol code — the MGD³-style latent guidance experiment in our Stage 4
was abandoned and removed on 2026-04-18.)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from cspd_eval.models.convnet import ConvNet
from cspd_eval.models.resnet import ResNet
from cspd_eval.models.resnet_ap import ResNetAP
from cspd_eval.train_utils import (
    AverageMeter,
    accuracy,
    random_indices,
    rand_bbox,
    Lighting,
    ColorJitter,
)

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# PCA lighting eigenvectors/eigenvalues (ImageNet)
IMAGENET_PCA_EIGVAL = [0.2175, 0.0188, 0.0045]
IMAGENET_PCA_EIGVEC = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]


def ipc_epoch(ipc: int, factor: int = 1, nclass: int = 10) -> int:
    """Compute training epochs based on IPC, matching the MGD³ reference eval."""
    effective_ipc = ipc * factor ** 2
    if effective_ipc == 1:
        epoch = 3000
    elif effective_ipc <= 10:
        epoch = 2000
    elif effective_ipc <= 50:
        epoch = 1500
    elif effective_ipc <= 200:
        epoch = 1000
    elif effective_ipc <= 500:
        epoch = 500
    else:
        epoch = 300

    if nclass == 100:
        epoch = int((2 / 3) * epoch)
    epoch = epoch - (epoch % 100)
    return epoch


def build_model(arch: str, nclass: int, size: int = 224, nch: int = 3) -> nn.Module:
    """Build evaluation model by architecture name."""
    if arch == "convnet":
        return ConvNet(
            num_classes=nclass,
            net_norm="instance",
            net_depth=6,
            net_width=128,
            channel=nch,
            im_size=(size, size),
        )
    elif arch == "resnet18":
        return ResNet(
            dataset="imagenet",
            depth=18,
            num_classes=nclass,
            norm_type="instance",
            size=size,
            nch=nch,
        )
    elif arch == "resnet_ap":
        return ResNetAP(
            dataset="imagenet",
            depth=10,
            num_classes=nclass,
            norm_type="instance",
            size=size,
            nch=nch,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}. Use convnet, resnet18, or resnet_ap.")


def build_train_transform(size: int) -> transforms.Compose:
    """ImageNet train transform with RRC + augmentation (matching the MGD³ eval protocol).

    Order: RRC → ToTensor → HFlip → ColorJitter(custom, tensor-space) → Lighting → Normalize.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.5, 1.0)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        Lighting(0.1, IMAGENET_PCA_EIGVAL, IMAGENET_PCA_EIGVEC),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_val_transform(size: int) -> transforms.Compose:
    """ImageNet validation transform (matching the MGD³ reference eval)."""
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    nclass: int,
    epoch: int,
    epochs: int,
    beta: float = 1.0,
    mix_p: float = 1.0,
) -> tuple[float, float, float]:
    """Train one epoch with CutMix augmentation."""
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()

    for input, target in loader:
        input = input.cuda()
        target = target.cuda()

        r = np.random.rand(1)
        if r < mix_p:
            # CutMix
            lam = np.random.beta(beta, beta)
            rand_index = random_indices(target, nclass=nclass)
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            output = model(input)
            loss = criterion(output, target) * ratio + criterion(output, target_b) * (1.0 - ratio)
        else:
            output = model(input)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return top1.avg, top5.avg, losses.avg


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float, float]:
    """Validate on the real test set."""
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    for input, target in loader:
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

    return top1.avg, top5.avg, losses.avg


def run_evaluation(
    *,
    distilled_dir: str | Path,
    val_dir: str | Path,
    arch: str = "convnet",
    nclass: int = 10,
    ipc: int = 10,
    size: int = 224,
    batch_size: int = 64,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    seed: int = 0,
    repeat: int = 1,
    num_workers: int = 4,
    save_dir: str | Path | None = None,
    epochs: int | None = None,
) -> dict[str, Any]:
    """Train a classifier on the distilled dataset and evaluate on real val set.

    Args:
        distilled_dir: Path to distilled dataset (ImageFolder format).
        val_dir: Path to real validation dataset (ImageFolder format).
        arch: Model architecture (convnet, resnet18, resnet_ap).
        nclass: Number of classes.
        ipc: Images per class in the distilled dataset.
        size: Image resolution.
        batch_size: Training batch size.
        lr: Learning rate.
        momentum: SGD momentum.
        weight_decay: Weight decay.
        seed: Random seed.
        repeat: Number of independent runs to average.
        num_workers: DataLoader workers.
        save_dir: Optional directory to save results.
        epochs: Override auto-computed epochs.

    Returns:
        Dict with best_acc1, best_acc5, all_results per repeat.
    """
    if seed >= 0:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    cudnn.benchmark = True

    # Auto-compute epochs from IPC if not overridden
    if epochs is None:
        epochs = ipc_epoch(ipc, factor=1, nclass=nclass)

    epoch_print_freq = max(epochs // 100, 1)

    # Build data loaders
    train_transform = build_train_transform(size)
    val_transform = build_val_transform(size)

    train_dataset = torchvision.datasets.ImageFolder(str(distilled_dir), transform=train_transform)
    val_dataset = torchvision.datasets.ImageFolder(str(val_dir), transform=val_transform)

    # Verify class count
    actual_nclass = len(train_dataset.classes)
    if actual_nclass != nclass:
        print(f"[WARN] Expected {nclass} classes but found {actual_nclass} in distilled dataset. Using {actual_nclass}.")
        nclass = actual_nclass

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size // 2, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )

    print(f"[Eval] arch={arch}, nclass={nclass}, ipc={ipc}, epochs={epochs}, size={size}")
    print(f"[Eval] distilled: {len(train_dataset)} images, val: {len(val_dataset)} images")
    print(f"[Eval] lr={lr}, momentum={momentum}, wd={weight_decay}, batch_size={batch_size}")
    print(f"[Eval] CutMix enabled (beta=1.0, mix_p=1.0)")
    print(f"[Eval] Repeat: {repeat}")

    all_best_acc1 = []
    all_best_acc5 = []
    all_results = []

    for run_idx in range(repeat):
        print(f"\n--- Run {run_idx + 1}/{repeat} ---")

        model = build_model(arch, nclass, size).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[2 * epochs // 3, 5 * epochs // 6], gamma=0.2,
        )

        best_acc1 = 0.0
        best_acc5 = 0.0
        run_log = []

        for epoch in tqdm(range(1, epochs + 1), desc=f"Run {run_idx+1}"):
            acc1_tr, acc5_tr, loss_tr = train_epoch(
                model, train_loader, criterion, optimizer, nclass,
                epoch, epochs,
            )

            if epoch % epoch_print_freq == 0 or epoch == epochs:
                acc1_val, acc5_val, loss_val = validate(model, val_loader, criterion)

                if acc1_val > best_acc1:
                    best_acc1 = acc1_val
                    best_acc5 = acc5_val

                run_log.append({
                    "epoch": epoch,
                    "train_acc1": round(acc1_tr, 2),
                    "val_acc1": round(acc1_val, 2),
                    "val_acc5": round(acc5_val, 2),
                    "best_acc1": round(best_acc1, 2),
                    "train_loss": round(loss_tr, 4),
                    "val_loss": round(loss_val, 4),
                })

                tqdm.write(
                    f"  Epoch {epoch}/{epochs} | "
                    f"Train: {acc1_tr:.1f}% | "
                    f"Val: {acc1_val:.1f}% (best: {best_acc1:.1f}%) | "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}"
                )

            scheduler.step()

        all_best_acc1.append(best_acc1)
        all_best_acc5.append(best_acc5)
        all_results.append({
            "run": run_idx,
            "best_acc1": round(best_acc1, 2),
            "best_acc5": round(best_acc5, 2),
            "log": run_log,
        })
        print(f"  Run {run_idx+1} best: {best_acc1:.1f}% (top-5: {best_acc5:.1f}%)")

    mean_acc1 = float(np.mean(all_best_acc1))
    std_acc1 = float(np.std(all_best_acc1))
    mean_acc5 = float(np.mean(all_best_acc5))

    print(f"\n{'='*60}")
    print(f"[Result] {arch} | {nclass} classes | IPC {ipc}")
    print(f"[Result] Best Top-1: {mean_acc1:.1f} +/- {std_acc1:.1f}")
    print(f"[Result] Best Top-5: {mean_acc5:.1f}")
    print(f"{'='*60}")

    result = {
        "arch": arch,
        "nclass": nclass,
        "ipc": ipc,
        "epochs": epochs,
        "size": size,
        "lr": lr,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "repeat": repeat,
        "seed": seed,
        "mean_best_acc1": round(mean_acc1, 2),
        "std_best_acc1": round(std_acc1, 2),
        "mean_best_acc5": round(mean_acc5, 2),
        "distilled_dir": str(distilled_dir),
        "val_dir": str(val_dir),
        "runs": all_results,
    }

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        result_path = save_dir / f"eval_{arch}.json"
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[Eval] Results saved to {result_path}")

    return result
