"""Eval training utilities: metrics, CutMix helpers, and the custom
tensor-space augmentation stack (Lighting + ColorJitter) required to match
the reference distillation-eval protocol.

Original augmentation code adapted from
https://github.com/eladhoffer/convNet.pytorch/blob/master/preprocess.py
"""

import random

import numpy as np
import torch


__all__ = [
    "AverageMeter",
    "accuracy",
    "random_indices",
    "rand_bbox",
    "Compose",
    "Lighting",
    "ColorJitter",
]


def accuracy(output, target, topk=(1,)):
    """Precision@k for the specified k values."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter:
    """Computes and stores the running average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def random_indices(y, nclass=10, intraclass=False, device="cuda"):
    n = len(y)
    if intraclass:
        index = torch.arange(n).to(device)
        for c in range(nclass):
            index_c = index[y == c]
            if len(index_c) > 0:
                randidx = torch.randperm(len(index_c))
                index[y == c] = index_c[randidx]
    else:
        index = torch.randperm(n).to(device)
    return index


def rand_bbox(size, lam):
    """CutMix bounding box sampler."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Lighting:
    """AlexNet-style PCA lighting noise."""

    def __init__(self, alphastd, eigval, eigvec, device="cpu"):
        self.alphastd = alphastd
        self.eigval = torch.tensor(eigval, device=device)
        self.eigvec = torch.tensor(eigvec, device=device)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = (
            self.eigvec.type_as(img).clone()
            .mul(alpha.view(1, 3).expand(3, 3))
            .mul(self.eigval.view(1, 3).expand(3, 3))
            .sum(1)
            .squeeze()
        )

        if len(img.shape) == 4:
            return img + rgb.view(1, 3, 1, 1).expand_as(img)
        return img + rgb.view(3, 1, 1).expand_as(img)


class _Grayscale:
    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class _Saturation:
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = _Grayscale()(img)
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class _Brightness:
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class _Contrast:
    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = _Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class ColorJitter:
    """Tensor-space ColorJitter (randomized order). Works on CHW tensors so
    it composes after ToTensor, unlike torchvision's PIL-space ColorJitter.
    """

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img):
        transforms = []
        if self.brightness != 0:
            transforms.append(_Brightness(self.brightness))
        if self.contrast != 0:
            transforms.append(_Contrast(self.contrast))
        if self.saturation != 0:
            transforms.append(_Saturation(self.saturation))

        random.shuffle(transforms)
        return Compose(transforms)(img)
