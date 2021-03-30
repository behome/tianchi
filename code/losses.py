#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/28 7:01 下午
# @Author  : 宋继贤
# @Description  : 
# @File    : losses.py
# @Software: PyCharm

import torch
import torch.nn as nn


class MultiBceLoss(nn.Module):

    def __init__(self):
        super(MultiBceLoss, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, pre, label):
        assert pre.shape == label.shape
        return self.loss(pre, label)
