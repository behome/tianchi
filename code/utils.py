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


def load_embedding(w2v_file, id2w, embedding_size=256):
    w2v = gensim.models.Word2Vec.load(w2v_file).wv
    embedding = torch.zeros((len(id2w), embedding_size), dtype=torch.float32)
    for key in id2w.keys():
        if key == 1:
            embedding[key] = torch.rand(embedding_size)
        elif key == 0:
            continue
        else:
            embedding[key] = torch.from_numpy(w2v[id2w[key]].copy())
    return embedding


def load_vocab(vocab_path, vocab_size):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    w2id = {'<pad>': 0, '<unk>': 1}
    id2w = {0: '<pad>', 1: '<unk>'}
    for i, line in enumerate(lines, 2):
        if i >= vocab_size:
            break
        word = line.strip('\n').split('\t')[0]
        w2id[word] = i
        id2w[i] = word
    return w2id, id2w


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
