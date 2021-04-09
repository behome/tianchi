#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/8 10:20 下午
# @Author  : 宋继贤
# @Description  : 
# @File    : train_ml.py
# @Software: PyCharm

import argparse
import numpy as np
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from extract_feature import SimpleFeature


class DataLoader:

    def __init__(self, data_path, simple_feature, class_id, sample=False):
        self.data_path = []
        self.simple_feature = simple_feature
        self.class_id = class_id
        self.sample = sample
        with open(data_path, 'r') as fin:
            lines = fin.readlines()
        self.positive_corpus = []
        self.negative_corpus = []
        for line in lines:
            report_id, txt, classes = line.strip('\n').split('|,|')
            classes = [int(class_id) for class_id in classes.split(' ') if len(class_id.strip()) > 0]
            if class_id in classes:
                self.positive_corpus.append((simple_feature.get_tf_idf([txt]).reshape(-1), 1))
            else:
                self.negative_corpus.append((simple_feature.get_tf_idf([txt]).reshape(-1), 0))
        if self.sample:
            self.slice_num = len(self.negative_corpus) // len(self.positive_corpus)
        else:
            self.slice_num = 1

    def get_subset(self, index=0):
        if index >= self.slice_num:
            raise ValueError("The index is out of range")
        if self.sample:
            corpus = self.positive_corpus + self.negative_corpus[index * len(self.positive_corpus):
                                                                 (index + 1) * len(self.positive_corpus)]
        else:
            corpus = self.positive_corpus + self.negative_corpus
        np.random.shuffle(corpus)
        return zip(*corpus)


def get_loss(y_pre, y_val):
    loss = log_loss(y_val, y_pre)
    return loss


def get_model(args):
    if args.model_type == 'lr':
        model = LogisticRegression()
    elif args.model_type == 'svm':
        # 'linear','rbf','sigmoid'
        model = SVC(kernel=args.kernel, probability=True)
    elif args.model_type == 'rfc':
        model = RandomForestClassifier()
    elif args.model_type == 'gdbt':
        model = GradientBoostingClassifier()
    else:
        raise ValueError("The %s model has not been implemented" % args.model_type)
    return model


def train(args):
    simple_feature = SimpleFeature(resume=True, cbow_path=args.cbow_path, tfidf_path=args.tfidf_path)
    train_data = DataLoader(args.train_data, simple_feature, args.class_id, sample=False)
    val_data = DataLoader(args.val_data, simple_feature, args.class_id, sample=False)
    X_val, y_val = val_data.get_subset()
    model_corpus = []
    result_sum = np.zeros((len(X_val)))
    for i in range(train_data.slice_num):
        X_train, y_train = train_data.get_subset(i)
        model = get_model(args)
        print("the corpus size is %d" % len(X_train))
        model.fit(X_train, y_train)
        result = model.predict_proba(X_val)
        result_sum += np.array(result[:, 1])
        loss = get_loss(result[:, 1], y_val)
        print("The %d th model loss is %.5f" % (i, loss))
        model_corpus.append(model)
    loss = get_loss(result_sum / len(model_corpus), y_val)
    print("The %s model at %d class final loss is %.5f" % (args.model_type, args.class_id, loss))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cbow_path", type=str, default='./user_data/model_data/vocab.pkl')
    parser.add_argument('--tfidf_path', type=str, default='./user_data/model_data/tf_idf.pkl')
    parser.add_argument('--train_data', type=str, default='./tc_data/track1_round1_train_20210222_train.csv',
                        help='the path to the directory containing the train data.')
    parser.add_argument("--val_data", type=str, default='./tc_data/track1_round1_train_20210222_val.csv',
                        help='the path to the directory containing the validation data.')
    parser.add_argument("--class_id", type=int, default=0, help='the class id to train model')
    parser.add_argument("--model_type", type=str, default="svm", help='the model to train')
    parser.add_argument("--kernel", type=str, default='rbf', help='the kernel of the svm')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    train(args)
