import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from fmix_orig import sample_mask

device = "cuda" if torch.cuda.is_available() else "cpu"

# https://www.kaggle.com/virajbagal/mixup-cutmix-fmix-visualisations/data
# https://www.kaggle.com/ar2017/pytorch-efficientnet-train-aug-cutmix-fmix
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cutmix(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.random.beta(alpha, alpha)
    lam = np.clip(lam, 0.3, 0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()
    new_data[:, :, bby1:bby2, bbx1:bbx2] = data[indices, :, bby1:bby2, bbx1:bbx2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    targets = (target, shuffled_target, lam)

    return new_data, targets


def resizemix(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = np.random.beta(alpha, alpha)
    lam = np.clip(lam, 0.3, 0.4)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    new_data = data.clone()

    resize_data = F.interpolate(
        data[indices], (bby2 - bby1, bbx2 - bbx1), mode="nearest"
    )

    new_data[:, :, bby1:bby2, bbx1:bbx2] = resize_data
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))

    targets = (target, shuffled_target, lam)

    return new_data, targets


def fmix(data, target, alpha, decay_power, shape, max_soft=0.0, reformulate=False):
    """https://github.com/ecs-vlc/fmix"""
    lam, mask = sample_mask(alpha, decay_power, shape, max_soft, reformulate)
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]
    x1 = torch.from_numpy(mask).to(device) * data
    x2 = torch.from_numpy(1 - mask).to(device) * shuffled_data

    targets = (target, shuffled_target, lam)

    return (x1 + x2), targets


# https://www.kaggle.com/sachinprabhu/pytorch-resnet50-snapmix-train-pipeline
# https://github.com/Shaoli-Huang/SnapMix/blob/main/utils/mixmethod.py
def get_spm(data, target, model):
    """camのヒートマップ作成。出力層直前の出力に全結合層の重みかける"""
    imgsize = (data.size()[3], data.size()[2])
    bs = data.size(0)
    with torch.no_grad():
        output = model(data.float())

        # 出力層の重み取得
        names = [name for name, layer in model.net.named_modules()]
        if "fc" in names:
            clsw = model.net.fc
        elif "head.fc" in names:  # rexnet
            clsw = model.net.head.fc
        else:  # efficientnet
            clsw = model.net.classifier
        weight = clsw.weight.data
        bias = clsw.bias.data
        weight = weight.view(weight.size(0), weight.size(1), 1, 1)

        # 出力層直前の出力
        fms = model.feat_forward(data.float())

        # 予測スコア
        clslogit = None

        # 出力層直前の出力に重みかける
        out = F.conv2d(fms, weight, bias=bias)

        # ヒートマップ作成
        outmaps = []
        for i in range(bs):
            evimap = out[i, target[i]]
            outmaps.append(evimap)

        outmaps = torch.stack(outmaps)

        if imgsize is not None:
            outmaps = outmaps.view(outmaps.size(0), 1, outmaps.size(1), outmaps.size(2))
            outmaps = F.interpolate(
                outmaps, imgsize, mode="bilinear", align_corners=False
            )

        outmaps = outmaps.squeeze()

        for i in range(bs):
            outmaps[i] -= outmaps[i].min()
            outmaps[i] /= outmaps[i].sum()

    return outmaps, clslogit


def snapmix(data, target, alpha=5.0, model=None):

    lam_a = torch.ones(data.size(0))
    lam_b = 1 - lam_a
    target_b = target.clone()

    wfmaps, _ = get_spm(data, target, model)
    bs = data.size(0)
    lam = np.random.beta(alpha, alpha)
    lam1 = np.random.beta(alpha, alpha)

    rand_index = torch.randperm(bs).to(device)
    wfmaps_b = wfmaps[rand_index, :, :]
    target_b = target[rand_index]

    same_label = target == target_b
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    bbx1_1, bby1_1, bbx2_1, bby2_1 = rand_bbox(data.size(), lam1)

    area = (bby2 - bby1) * (bbx2 - bbx1)
    area1 = (bby2_1 - bby1_1) * (bbx2_1 - bbx1_1)

    if area1 > 0 and area > 0:
        ncont = data[rand_index, :, bbx1_1:bbx2_1, bby1_1:bby2_1].clone()
        ncont = F.interpolate(
            ncont, size=(bbx2 - bbx1, bby2 - bby1), mode="bilinear", align_corners=True
        )
        data[:, :, bbx1:bbx2, bby1:bby2] = ncont
        lam_a = 1 - wfmaps[:, bbx1:bbx2, bby1:bby2].sum(2).sum(1) / (
            wfmaps.sum(2).sum(1) + 1e-8
        )
        lam_b = wfmaps_b[:, bbx1_1:bbx2_1, bby1_1:bby2_1].sum(2).sum(1) / (
            wfmaps_b.sum(2).sum(1) + 1e-8
        )
        tmp = lam_a.clone()
        lam_a[same_label] += lam_b[same_label]
        lam_b[same_label] += tmp[same_label]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
        lam_a[torch.isnan(lam_a)] = lam
        lam_b[torch.isnan(lam_b)] = 1 - lam

    targets = (target, target_b, lam_a.to(device), lam_b.to(device))

    return data, targets


class SnapMixLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, criterion, outputs, ya, yb, lam_a, lam_b):
        loss_a = criterion(outputs, ya)
        loss_b = criterion(outputs, yb)
        loss = torch.mean(loss_a * lam_a + loss_b * lam_b)
        return loss
