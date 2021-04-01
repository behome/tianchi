#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/28 7:01 下午
# @Author  : 宋继贤
# @Description  : 
# @File    : losses.py
# @Software: PyCharm

import math
import numpy as np
import torch
import torch.nn as nn


class MultiBceLoss(nn.Module):

    def __init__(self):
        super(MultiBceLoss, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, pre, label):
        pre = pre.squeeze()
        assert pre.shape == label.shape
        return self.loss(pre, label)


class MultiWeightedBCELoss(nn.Module):

    def __init__(self, weight_classes, device, eps=1e-8):
        super(MultiWeightedBCELoss, self).__init__()
        self.weights_classes = torch.from_numpy(np.load(weight_classes)).cuda(device)
        self.eps = eps

    def forward(self, pre, label):
        assert pre.shape == label.shape
        positive = self.weights_classes[:, 1].unsqueeze(0) * label * torch.log(pre + self.eps)
        negative = self.weights_classes[:, 0].unsqueeze(0) * (1 - label) * torch.log(1 - pre + self.eps)
        loss = -(positive + negative).mean()
        assert (not math.isnan(loss.item())
                and not math.isinf(loss.item()))
        return loss
