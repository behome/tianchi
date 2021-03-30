#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/29 10:32 上午
# @Author  : 宋继贤
# @Description  : 
# @File    : test.py
# @Software: PyCharm

import os
import sys
import json
import argparse
import torch
import numpy as np
sys.path.append(os.path.curdir)
import models
from data_op import MedicalTestDataloader


def test(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    test_data = MedicalTestDataloader(args.test_data, args.batch_size, args.num_workers)
    if args.model_type == 'lstm':
        model = models.LSTMModel(args)
    elif args.model_type == 'conv':
        model = models.ConvModel(args)
    else:
        raise NotImplementedError
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if not os.path.isfile(os.path.join(args.checkpoint_path, "model.pth")):
        print("No model to test")
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, "model.pth")))
    model = model.cuda(args.device)
    model.eval()
    results = []
    for i, data in enumerate(test_data):
        tmp = [_.cuda(args.device) if isinstance(_, torch.Tensor) else _ for _ in data]
        report_ids, sentence_ids, sentence_lengths = tmp
        pre = model(sentence_ids, sentence_lengths)
        results.extend(zip(report_ids, pre.cpu().detach().numpy().tolist()))
    with open(os.path.join(args.checkpoint_path, "result.csv"), 'w', encoding='utf-8') as res_out:
        for item in results:
            vector = list(map(lambda x: "%s" % x, item[1]))
            txt = '|,|'.join([item[0], ' '.join(vector)]) + "\n"
            res_out.write(txt)
    print("The test finished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./user_data/model_data/config.json', help='the config file about the model')
    x = parser.parse_args()
    with open(x.config, 'r') as fin:
        config = json.load(fin)
    ns = argparse.Namespace(**config)
    test(ns)
