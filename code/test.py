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
from data_op import MedicalTestDataloader, MedicalEasyEnsembleDataloader, MedicalDataloader
from losses import MultiBceLoss
from extract_feature import SimpleFeature
import utils


def test(global_args):
    torch.manual_seed(global_args.seed)
    np.random.seed(global_args.seed)
    if global_args.use_tfidf:
        simple_feature = SimpleFeature(resume=True, vocab_path=global_args.vocab_path, tfidf_path=global_args.tfidf_path)
    else:
        simple_feature = None
    test_data = MedicalTestDataloader(global_args.test_data, global_args.batch_size, global_args.num_workers,
                                      use_tfidf=global_args.use_tfidf, simple_feature=simple_feature)
    model_corpus = load_all_model(global_args.root_dir, global_args.device)

    results = []
    for i, data in enumerate(test_data):
        tmp = [_.cuda(global_args.device) if isinstance(_, torch.Tensor) else _ for _ in data]
        report_ids, sentence_ids, sentence_lengths = tmp
        pre_vectors = []
        for class_id in range(len(model_corpus)):
            pre_cur_sum = np.zeros((sentence_ids.shape[0], 1), dtype=np.float32)
            for j in range(len(model_corpus[class_id])):
                pre = model_corpus[class_id][j](sentence_ids, sentence_lengths)
                pre_cur_sum += pre.cpu().detach().numpy()
            pre_vectors.append(pre_cur_sum / len(model_corpus[class_id]))
        results.extend(zip(report_ids, np.concatenate(pre_vectors, axis=1).tolist()))
    with open(os.path.join(global_args.root_dir, "result.csv"), 'w', encoding='utf-8') as res_out:
        for item in results:
            vector = list(map(lambda x: "%s" % x, item[1]))
            txt = '|,|'.join([item[0], ' '.join(vector)]) + "\n"
            res_out.write(txt)
    print("The test finished")


def eval_model(global_args):
    torch.manual_seed(global_args.seed)
    np.random.seed(global_args.seed)
    if global_args.use_tfidf:
        simple_feature = SimpleFeature(resume=True, vocab_path=global_args.vocab_path,
                                       tfidf_path=global_args.tfidf_path)
    else:
        simple_feature = None
    val_data = MedicalEasyEnsembleDataloader(global_args.val_data, global_args.class_id, global_args.batch_size,
                                             False,
                                             global_args.num_workers,
                                             use_tfidf=global_args.use_tfidf, simple_feature=simple_feature)
    loss_func = MultiBceLoss()
    model_corpus = load_class_model(global_args.root_dir, global_args.class_id, global_args.device)
    loss_sum = 0
    val_num = 0
    for i, data in enumerate(val_data):
        tmp = [_.cuda(global_args.device) if isinstance(_, torch.Tensor) else _ for _ in data]
        report_ids, sentence_ids, sentence_lengths, class_vec = tmp
        pre_cur_sum = torch.zeros((sentence_ids.shape[0], 1), dtype=torch.float32).cuda(global_args.device)
        for j in range(len(model_corpus)):
            pre = model_corpus[j](sentence_ids, sentence_lengths)
            pre_cur_sum += pre
        pre_avg = pre_cur_sum / len(model_corpus)
        print(pre_avg.shape)
        loss = loss_func(pre_avg, class_vec)
        loss_sum += loss.item()
        print("val %d iter  loss %.3f\t" % (i, loss.item()))
        val_num += 1
    print("The Average loss is %.3f" % (loss_sum / val_num))


def load_all_model(root_dir, device=0):
    model_corpus = []
    for i in range(17):
        item = load_class_model(root_dir, i, device)
        model_corpus.append(item)
    return model_corpus


def load_class_model(root_dir, class_id, device=0):
    config_file = os.path.join(root_dir, str(class_id), "config.json")
    with open(config_file, 'r') as fin:
        config = json.load(fin)
    args = argparse.Namespace(**config)
    item = []
    for j in range(args.model_num):
        if args.model_type == 'lstm':
            model = models.LSTMModel(args)
        elif args.model_type == 'conv':
            model = models.ConvModel(args)
        elif args.model_type == 'char':
            model = models.CharCNNModel(args)
        elif args.model_type == 'base':
            model = models.BaseModel(args)
        else:
            raise NotImplementedError
        model_path = os.path.join(args.checkpoint_path, str(class_id), "%s_%s" % (args.model_type, args.type_suffix),
                                  "model_%d.pth" % j)
        if not os.path.isfile(model_path):
            print("No model to test")
            exit(1)
        model.load_state_dict(torch.load(model_path))
        model = model.cuda(device)
        model.eval()
        item.append(model)
    return item


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default='./user_data/model_data/',
                        help='the model root about the model')
    parser.add_argument("--test_data", type=str, default='./tc_data/track1_round1_testA_20210222.csv')
    parser.add_argument("--val_data", type=str, default='./tc_data/track1_round1_train_20210222_val.csv',
                        help='the path to the directory containing the validation data.')
    parser.add_argument("--batch_size", type=int, default=64, help="The batch size to predict")
    parser.add_argument("--num_workers", type=int, default=1, help="The number of dataloader processes")
    parser.add_argument("--seed", type=int, default=9233, help="The random seed to all")
    parser.add_argument("--device", type=int, default=0, help="The device to use")
    parser.add_argument("--use_tfidf", type=utils.str2bool, default=False, help='whether to use tf-idf feature')
    parser.add_argument("--vocab_path", type=str, default='./user_data/model_data/vocab.pkl',
                        help='the path store cbow model')
    parser.add_argument('--tfidf_path', type=str, default='./user_data/model_data/tf_idf.pkl',
                        help='the path store tfidf model')
    parser.add_argument('--test', type=utils.str2bool, default=True, help='whether to generate test result')
    parser.add_argument('--class_id', type=int, default=0, help='class to eval')
    x = parser.parse_args()
    if x.test:
        test(x)
    else:
        eval_model(x)
