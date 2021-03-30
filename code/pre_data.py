#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/27 8:16 下午
# @Author  : 宋继贤
# @Description  : 
# @File    : pre_data.py
# @Software: PyCharm


import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


def get_vocab(training_path):
    with open(training_path, 'r') as fin:
        lines = fin.readlines()
    max_ids = 0
    length_counter = Counter()
    for line in lines:
        report_id, txt, c = line.strip('\n').split('|,|')
        txt_ids = [int(ids) for ids in txt.split(' ') if ids.strip() != '']
        length_counter[len(txt_ids)] += 1
        max_ids_tmp = max(txt_ids)
        max_ids = max(max_ids_tmp, max_ids)
    plt.figure()
    num_keys = sorted(list(length_counter.keys()))
    plt.bar(x=num_keys, height=[length_counter.get(key) for key in num_keys])
    # plt.xticks(num_keys)
    plt.show()
    return max_ids


def show_classes(training_path):
    with open(training_path, 'r') as fin:
        lines = fin.readlines()
    classes_counter = Counter()
    for line in lines:
        report_id, txt, c = line.strip('\n').split('|,|')
        c_ids = [int(ids) for ids in c.split(' ') if ids.strip() != '']
        for c_id in c_ids:
            classes_counter[c_id] += 1
        if len(c_ids) == 0:
            classes_counter[17] += 1
    plt.figure()
    num_keys = sorted(list(classes_counter.keys()))
    plt.bar(x=num_keys, height=[classes_counter.get(key) for key in num_keys])
    plt.xticks(num_keys)
    plt.show()


def split_train_val(origin_path, val_num):
    with open(origin_path, 'r') as fin:
        lines = fin.readlines()
    assert len(lines) >= val_num
    num_data = len(lines)
    val_index = np.random.choice(num_data, val_num, replace=False)
    train_index = np.delete(np.arange(num_data), val_index)
    train_data = [lines[index] for index in train_index]
    val_data = [lines[index] for index in val_index]
    file_title = os.path.splitext(origin_path)[0]
    train_file_name = file_title + '_train.csv'
    val_file_name = file_title + "_val.csv"
    with open(train_file_name, 'w', encoding='utf-8') as fout:
        print("Write train data %d" % len(train_data))
        fout.writelines(train_data)
    with open(val_file_name, 'w', encoding='utf-8') as fout:
        print("Write val data %d" % len(val_data))
        fout.writelines(val_data)


if __name__ == '__main__':
    # max_id = get_vocab('../tc_data/track1_round1_train_20210222.csv')
    # print(max_id)
    show_classes('../tc_data/track1_round1_train_20210222.csv')
    # split_train_val('../tc_data/track1_round1_train_20210222.csv', 2000)

