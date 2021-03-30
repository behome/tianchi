#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/27 7:56 下午
# @Author  : 宋继贤
# @Description  : 
# @File    : train.py
# @Software: PyCharm


import os
import sys
import json
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
sys.path.append(os.path.curdir)
import models as models
from data_op import MedicalDataloader
from losses import MultiBceLoss
import utils as utils


def parse_args():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--train_data', type=str, default='./tc_data/track1_round1_train_20210222_train.csv',
                        help='the path to the directory containing the train data.')
    parser.add_argument("--val_data", type=str, default='./tc_data/track1_round1_train_20210222_val.csv',
                        help='the path to the directory containing the validation data.')
    parser.add_argument('--test_data', type=str, default='./tc_data/track1_round1_testA_20210222.csv',
                        help='the path to the directory containing the data.')
    parser.add_argument("--vocab_size", type=int, default=858, help="the size of the vocabulary")
    parser.add_argument("--embedding_size", type=int, default=256, help="the size of the word embedding")
    parser.add_argument("--hidden_size", type=int, default=256, help="the size of the hidden layer")
    parser.add_argument("--conv_hidden", type=int, default=100, help="the output channel of the conv")
    parser.add_argument("--output_classes", type=int, default=17, help="the classes of the output")
    parser.add_argument("--lstm_layer", type=int, default=2, help="the number of the lstm layer")
    parser.add_argument("--batch_size", type=int, default=64, help="the size of one batch")
    parser.add_argument("--dropout", type=float, default=0.5, help="the probability to set one unit to zero")
    parser.add_argument("--bn", type=int, default=0, help="option about whether to use bn layer")
    parser.add_argument("--checkpoint_path", type=str, default="./user_data/model_data/",
                        help="the path to save model and tensorboard data")
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate for the visual extractor.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument("--start_epoch", type=int, default=10, help="the epoch to start train")
    parser.add_argument("--decay_every", type=int, default=5, help='the lr will decay per ')
    parser.add_argument("--decay_rate", type=float, default=0.8, help="the rate to decay lr")
    parser.add_argument("--epochs", type=int, default=100, help="the epoch to train model")
    parser.add_argument("--num_workers", type=int, default=1, help="the number of the process to read data")
    parser.add_argument("--max_length", type=int, default=100, help="the max length of the report sentence")
    parser.add_argument("--mlp_units", type=list, default=[256, ], help="The hidden units of the output layer")
    parser.add_argument("--model_type", type=str, default="lstm", help="The type model to use")
    parser.add_argument("--device", type=int, default=0, help="the gpu device to run training")
    parser.add_argument("--print_every", type=int, default=20, help="the period to print loss")
    parser.add_argument("--save_every", type=int, default=100, help="the period to save model")
    parser.add_argument("--val_every", type=int, default=400, help="the period to validate model")
    parser.add_argument("--loss_log_every", type=int, default=200, help="the period to log loss")
    parser.add_argument("--patient", type=int, default=5, help='the patient to early stop')
    parser.add_argument("--bi", type=bool, default=True, help="whether to use bilstm")
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')
    args = parser.parse_args()
    return args


def eval_model(model, loss_func, val_data, epoch):
    model.eval()
    val_loss = 0
    val_num = 0
    for i, data in enumerate(val_data):
        tmp = [_.cuda(args.device) if isinstance(_, torch.Tensor) else _ for _ in data]
        report_ids, sentence_ids, sentence_lengths, output_vec = tmp
        loss = loss_func(model(sentence_ids, sentence_lengths), output_vec)
        val_num += 1
        print("val iter %d epoch %d loss: %.3f" % (i, epoch, loss.item()))
        val_loss += loss.item()
    model.train()
    return val_loss / val_num


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_data = MedicalDataloader(args.train_data, args.output_classes, args.batch_size, True, args.num_workers)
    val_data = MedicalDataloader(args.val_data, args.output_classes, args.batch_size, False, args.num_workers)
    if args.model_type == 'lstm':
        model = models.LSTMModel(args)
    elif args.model_type == 'conv':
        model = models.ConvModel(args)
    else:
        raise NotImplementedError
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if os.path.isfile(os.path.join(args.checkpoint_path, "model.pth")):
        print("Load model from previous step")
        model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, "model.pth")))
    tb = SummaryWriter(args.checkpoint_path)
    iteration = 0
    model = model.cuda(args.device)
    model.train()
    optimizer = utils.build_optimizer(args, model)
    loss_func = MultiBceLoss()
    cur_worse = 1000
    bad_times = 0
    for epoch in range(args.epochs):
        if epoch >= args.start_epoch:
            factor = (epoch - args.start_epoch) // args.decay_every
            decay_factor = args.decay_rate ** factor
            current_lr = args.lr * decay_factor
            utils.set_lr(optimizer, current_lr)
        for i, data in enumerate(train_data):
            tmp = [_.cuda(args.device) if isinstance(_, torch.Tensor) else _ for _ in data]
            report_ids, sentence_ids, sentence_lengths, output_vec = tmp
            optimizer.zero_grad()
            loss = loss_func(model(sentence_ids, sentence_lengths), output_vec)
            loss.backward()
            train_loss = loss.item()
            optimizer.step()
            iteration += 1
            if iteration % args.print_every == 0:
                print("iter %d epoch %d loss: %.3f" % (iteration, epoch, train_loss))

            if iteration % args.save_every == 0:
                torch.save(model.state_dict(), os.path.join(args.checkpoint_path, "model.pth"))
                with open(os.path.join(args.checkpoint_path, "config.json"), 'w', encoding='utf-8') as config_f:
                    json.dump(vars(args), config_f)
            if iteration % args.val_every == 0:
                val_loss = eval_model(model, loss_func, val_data, epoch)
                tb.add_scalar("val_loss", val_loss, iteration)
                if val_loss > cur_worse:
                    cur_worse = val_loss
                    bad_times += 1
                else:
                    cur_worse = val_loss
                    bad_times = 0
                if bad_times > args.patient:
                    print('Early Stop !!!!')
                    exit(0)
            if iteration % args.loss_log_every == 0:
                tb.add_scalar("train_loss", loss.item(), iteration)

    print("The train finished")


if __name__ == '__main__':
    args = parse_args()
    with open(os.path.join(args.checkpoint_path, "config.json"), 'w', encoding='utf-8') as config_f:
        json.dump(vars(args), config_f)
    # train(args)


