#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/28 7:01 下午
# @Author  : 宋继贤
# @Description  : 
# @File    : losses.py
# @Software: PyCharm

import numpy as np
import torch
import torch.nn as nn


class MultiBceLoss(nn.Module):

    def __init__(self):
        super(MultiBceLoss, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, pre, label):
        assert pre.shape == label.shape
        return self.loss(pre, label)


class MultiWeightedBCELoss(nn.Module):

    def __init__(self, weight_classes):
        super(MultiWeightedBCELoss, self).__init__()
        self.weights_classes = torch.from_numpy(np.load(weight_classes))

    def forward(self, pre, label):
        assert pre.shape == label.shape
        positive = self.weights_classes[:, 1].unsqueeze(0) * label * torch.log(pre)
        negative = self.weights_classes[:, 0].unsqueeze(0) * (1 - label) * torch.log(1 - pre)
        loss = -(positive + negative).mean()
        return loss
