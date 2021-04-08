#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/28 8:39 下午
# @Author  : 宋继贤
# @Description  : 
# @File    : utils.py
# @Software: PyCharm

import torch
import gensim
import argparse


def build_optimizer(args, param):
    optimizer = getattr(torch.optim, args.optim)(
        param,
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )
    return optimizer


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def load_embedding(w2v_file, vocab_size=859, embedding_size=256):
    w2v = gensim.models.Word2Vec.load(w2v_file).wv
    embedding = torch.zeros((vocab_size, embedding_size), dtype=torch.float32)
    for i in range(1, vocab_size):
        embedding[i] = torch.from_numpy(w2v[str(i - 1)].copy())
    return embedding


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
