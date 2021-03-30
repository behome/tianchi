#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/28 8:39 下午
# @Author  : 宋继贤
# @Description  : 
# @File    : utils.py
# @Software: PyCharm

import torch


def build_optimizer(args, model):
    optimizer = getattr(torch.optim, args.optim)(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )
    return optimizer


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr