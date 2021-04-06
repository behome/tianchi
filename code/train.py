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
from data_op import MedicalDataloader, MedicalEasyEnsembleDataloader
from losses import MultiWeightedBCELoss, MultiBceLoss
import utils as utils


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--train_data', type=str, default='./tc_data/track1_round1_train_20210222_train.csv',
                        help='the path to the directory containing the train data.')
    parser.add_argument("--val_data", type=str, default='./tc_data/track1_round1_train_20210222_val.csv',
                        help='the path to the directory containing the validation data.')
    parser.add_argument('--test_data', type=str, default='./tc_data/track1_round1_testA_20210222.csv',
                        help='the path to the directory containing the data.')
    parser.add_argument('--class_weight', type=str, default='./user_data/statistical_data/classes_weight.npy',
                        help="the weight about every class")
    parser.add_argument('--co_occur', type=str, default='./user_data/statistical_data/co_occur_norm.npy',
                        help='the occurrence matrix')
    parser.add_argument('--fix_occur', type=str2bool, default=False, help='whether to train occurrence matrix')
    parser.add_argument('--multi_lambda', type=float, default=0.2, help='the weight of the multi-bce loss')
    parser.add_argument('--single_lambda', type=float, default=0.8, help='the weight of the single-bce loss')
    parser.add_argument('--start_train_occur', type=int, default=5, help='the epoch to start train occurrence matrix')
    parser.add_argument("--class_id", type=int, default=0, help='the class id to train model')
    parser.add_argument("--model_num", type=int, default=4, help='the number model will be trained for current class')
    parser.add_argument("--vocab_size", type=int, default=859, help="the size of the vocabulary")
    parser.add_argument("--embedding_size", type=int, default=256, help="the size of the word embedding")
    parser.add_argument("--w2v_file", type=str, default='./user_data/model_data/word2vec.256d.858.bin',
                        help='pretrained embedding')
    parser.add_argument("--hidden_size", type=int, default=256, help="the size of the hidden layer")
    parser.add_argument("--conv_hidden", type=int, default=100, help="the output channel of the conv")
    parser.add_argument("--output_classes", type=int, default=1, help="the classes of the output")
    parser.add_argument("--lstm_layer", type=int, default=1, help="the number of the lstm layer")
    parser.add_argument("--batch_size", type=int, default=128, help="the size of one batch")
    parser.add_argument("--dropout", type=float, default=0.5, help="the probability to set one unit to zero")
    parser.add_argument("--bn", type=int, default=0, help="option about whether to use bn layer")
    parser.add_argument("--checkpoint_path", type=str, default="./user_data/occ_model/",
                        help="the path to save model and tensorboard data")
    parser.add_argument("--sample_every", type=int, default=30, help='the period to resample negative data')
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr', type=float, default=1e-3, help='the learning rate for the visual extractor.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='the weight decay.')
    parser.add_argument("--start_epoch", type=int, default=10, help="the epoch to start decay lr")
    parser.add_argument("--decay_every", type=int, default=5, help='the lr will decay per ')
    parser.add_argument("--decay_rate", type=float, default=0.8, help="the rate to decay lr")
    parser.add_argument("--epochs", type=int, default=100, help="the epoch to train model")
    parser.add_argument("--num_workers", type=int, default=1, help="the number of the process to read data")
    parser.add_argument("--max_length", type=int, default=100, help="the max length of the report sentence")
    parser.add_argument("--mlp_units", type=list, default=[], help="The hidden units of the output layer")
    parser.add_argument("--model_type", type=str, default="char", help="The type model to use")
    parser.add_argument("--type_suffix", type=str, default="",
                        help="a suffix about this model type to distinguish each training")
    parser.add_argument("--device", type=int, default=0, help="the gpu device to run training")
    parser.add_argument("--print_every", type=int, default=20, help="the period to print loss")
    parser.add_argument("--save_every", type=int, default=100, help="the period to save model")
    parser.add_argument("--val_every", type=int, default=100, help="the period to validate model")
    parser.add_argument("--loss_log_every", type=int, default=200, help="the period to log loss")
    parser.add_argument("--patient", type=int, default=5, help='the patient to early stop')
    parser.add_argument("--threshold", type=int, default=0.5,
                        help='the threshold to determine the positive and negative')
    parser.add_argument("--bi", type=str2bool, default=True, help="whether to use bilstm")
    parser.add_argument('--amsgrad', type=str2bool, default=True, help='.')
    args = parser.parse_args()
    return args


def eval_model(args, model, multi_loss_func, single_loss_func, val_data, epoch):
    model.eval()
    val_loss = 0
    val_num = 0
    for i, data in enumerate(val_data):
        tmp = [_.cuda(args.device) if isinstance(_, torch.Tensor) else _ for _ in data]
        report_ids, sentence_ids, sentence_lengths, single_vec, class_vec = tmp
        multi_pro, pre = model(sentence_ids, sentence_lengths)
        multi_loss = multi_loss_func(multi_pro, class_vec)
        single_loss = single_loss_func(pre, single_vec)
        loss = args.multi_lambda * multi_loss + args.single_lambda * single_loss
        print("val %d iter %d epoch:\t loss %.3f\t single loss %.3f \t multi loss %.3f" %
              (i, epoch, loss.item(), single_loss.item(), multi_loss.item()))
        val_num += 1
        val_loss += loss.item()
    model.train()
    return val_loss / val_num


def train(args, model_id, tb):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_data = MedicalEasyEnsembleDataloader(args.train_data, args.class_id, args.batch_size, True, args.num_workers)
    val_data = MedicalEasyEnsembleDataloader(args.val_data, args.class_id, args.batch_size, False, args.num_workers)
    if os.path.exists(args.w2v_file):
        print("Loading pretrained embedding")
        embedding = utils.load_embedding(args.w2v_file, vocab_size=args.vocab_size, embedding_size=args.embedding_size)
    else:
        embedding = None
    if os.path.exists(args.co_occur):
        print("Loading occurrence matrix")
        occurrence = torch.from_numpy(np.load(args.co_occur)[:, args.class_id]).unsqueeze(0)
    else:
        occurrence = None
    if args.model_type == 'lstm':
        model = models.LSTMModel(args, embedding, occurrence)
    elif args.model_type == 'conv':
        model = models.ConvModel(args, embedding, occurrence)
    elif args.model_type == 'char':
        model = models.CharCNNModel(args, embedding, occurrence)
    elif args.model_type == 'base':
        model = models.BaseModel(args, embedding, occurrence)
    else:
        raise NotImplementedError
    if os.path.isfile(os.path.join(args.checkpoint_path, str(args.class_id),
                                   "%s_%s" % (args.model_type, args.type_suffix), "model_%d.pth" % model_id)):
        print("Load %d class %s type %dth model from previous step" % (args.class_id, args.model_type, model_id))
        model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, str(args.class_id),
                                                      "%s_%s" % (args.model_type, args.type_suffix),
                                                      "model_%d.pth" % model_id)))
    iteration = 0
    model = model.cuda(args.device)
    model.train()
    co_param = list(map(id, model.co_occur_layer.parameters()))
    other_param = filter(lambda x: id(x) not in co_param, model.parameters())
    other_optimizer = utils.build_optimizer(args, other_param)
    co_optimizer = utils.build_optimizer(args, model.co_occur_layer.parameters())
    single_loss_func = MultiBceLoss()
    multi_loss_func = MultiBceLoss()
    cur_worse = 1000
    bad_times = 0
    for epoch in range(args.epochs):
        if epoch >= args.start_epoch:
            factor = (epoch - args.start_epoch) // args.decay_every
            decay_factor = args.decay_rate ** factor
            current_lr = args.lr * decay_factor
            utils.set_lr(other_optimizer, current_lr)
            utils.set_lr(co_optimizer, current_lr)
        # if epoch != 0 and epoch % args.sample_every == 0:
        #     train_data.re_sample()
        for i, data in enumerate(train_data):
            tmp = [_.cuda(args.device) if isinstance(_, torch.Tensor) else _ for _ in data]
            report_ids, sentence_ids, sentence_lengths, single_vec, class_vec = tmp
            other_optimizer.zero_grad()
            co_optimizer.zero_grad()
            multi_pro, pre = model(sentence_ids, sentence_lengths)
            multi_loss = multi_loss_func(multi_pro, class_vec)
            single_loss = single_loss_func(pre, single_vec)
            loss = args.multi_lambda * multi_loss + args.single_lambda * single_loss
            loss.backward()
            train_loss = loss.item()
            other_optimizer.step()
            if epoch >= args.start_train_occur:
                co_optimizer.step()
            iteration += 1
            if iteration % args.print_every == 0:
                print("iter %d epoch %d loss: %.3f" % (iteration, epoch, train_loss))

            if iteration % args.save_every == 0:
                torch.save(model.state_dict(), os.path.join(args.checkpoint_path, str(args.class_id),
                                                            "%s_%s" % (args.model_type, args.type_suffix),
                                                            "model_%d.pth" % model_id))
                with open(os.path.join(args.checkpoint_path, str(args.class_id), "config.json"), 'w', encoding='utf-8') as config_f:
                    json.dump(vars(args), config_f, indent=2)
                with open(os.path.join(args.checkpoint_path, str(args.class_id),
                                       "%s_%s" % (args.model_type, args.type_suffix), "config.json"), 'w',
                          encoding='utf-8') as config_f:
                    json.dump(vars(args), config_f, indent=2)
            if iteration % args.val_every == 0:
                val_loss = eval_model(args, model, multi_loss_func, single_loss_func, val_data, epoch)
                tb.add_scalar("model_%d val_loss" % model_id, val_loss, iteration)
                if val_loss > cur_worse:
                    print("Bad Time Appear")
                    cur_worse = val_loss
                    bad_times += 1
                else:
                    cur_worse = val_loss
                    bad_times = 0
                if bad_times > args.patient:
                    print('Early Stop !!!!')
                    return
            if iteration % args.loss_log_every == 0:
                tb.add_scalar("model_%d train_loss" % model_id, loss.item(), iteration)

    print("The train finished")


if __name__ == '__main__':
    args = parse_args()
    # with open(os.path.join(args.checkpoint_path, "config.json"), 'w', encoding='utf-8') as config_f:
    #     json.dump(vars(args), config_f)
    if not os.path.exists(os.path.join(args.checkpoint_path, str(args.class_id),
                                       "%s_%s" % (args.model_type, args.type_suffix),)):
        os.makedirs(os.path.join(args.checkpoint_path, str(args.class_id),
                                 "%s_%s" % (args.model_type, args.type_suffix),))
    sw = SummaryWriter(os.path.join(args.checkpoint_path, str(args.class_id),
                                    "%s_%s" % (args.model_type, args.type_suffix),))
    for i in range(args.model_num):
        print("==========================Start training the %d class %dth model===============" % (args.class_id, i))
        train(args, i, sw)


