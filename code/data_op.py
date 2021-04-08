#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/27 7:57 下午
# @Author  : 宋继贤
# @Description  : 
# @File    : data_op.py
# @Software: PyCharm

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import load_vocab


class MedicalDataset(Dataset):

    def __init__(self, data_path, vocab_path, vocab_size, class_id, num_class=17, max_length=100, sample=True,
                 use_tfidf=False, simple_feature=None):
        super(MedicalDataset, self).__init__()
        with open(data_path, 'r') as fin:
            lines = fin.readlines()
        self.corpus = []
        self.use_tfidf = use_tfidf
        self.class_id = class_id
        self.sample = sample
        self.w2id, self.id2w = load_vocab(vocab_path, vocab_size)
        if self.use_tfidf and simple_feature is None:
            raise ValueError("the simple feature object can not be None when using tf-idf")
        self.simple_feature = simple_feature
        self.positive_corpus = []
        self.negative_corpus = []
        for line in lines:
            report_id, txt, classes = line.strip('\n').split('|,|')
            if use_tfidf:
                txt_ids = txt
            else:
                txt_ids = [self.w2id.get(ids, self.w2id['<unk>']) for ids in txt.split(' ') if len(ids.strip()) > 0]
            classes = [int(class_id) for class_id in classes.split(' ') if len(class_id.strip()) > 0]
            if self.class_id in classes:
                self.positive_corpus.append({'report_id': report_id, 'txt': txt_ids, 'class_label': 1, 'label': 1,
                                             'classes': classes})
            else:
                if len(classes) == 0:
                    item = {'report_id': report_id, 'txt': txt_ids, 'class_label': 1, 'label': 0,
                                             'classes': classes}
                else:
                    item = {'report_id': report_id, 'txt': txt_ids, 'class_label': 1, 'label': 1,
                                             'classes': classes}
                self.negative_corpus.append(item)
        if self.class_id == 17:
            self.slice_num = 1
        else:
            self.slice_num = len(self.negative_corpus) // len(self.positive_corpus)
        self.corpus = []
        self.re_sample()
        self.num_class = num_class
        self.max_length = max_length

    def __getitem__(self, index):
        item = self.corpus[index]
        report_id = item['report_id']
        report_txt = item['txt']
        if self.use_tfidf:
            report_length = len(report_txt.split())
            report_txt = self.simple_feature.get_tf_idf([report_txt]).reshape(-1)
        else:
            report_length = len(report_txt)
        classes = item['classes']
        class_label = item['class_label']
        label = item['label']
        class_vec = np.zeros((17,))
        for class_id in classes:
            class_vec[class_id] = 1.0
        return report_id, report_txt, report_length, class_vec, class_label, label

    def __len__(self):
        return len(self.corpus)

    def re_sample(self, slice_index=0):
        if self.sample == 'random':
            sample_negative_index = np.random.choice(len(self.negative_corpus), len(self.positive_corpus),
                                                     replace=False)
            sample_negative_corpus = [self.negative_corpus[index] for index in sample_negative_index]
            self.corpus = self.positive_corpus + sample_negative_corpus
        elif self.sample == 'order':
            self.corpus = self.positive_corpus + self.negative_corpus[slice_index * len(self.positive_corpus):]
        else:
            self.corpus = self.positive_corpus + self.negative_corpus
        print("++++++++After sampling total number of corpus is %d+++++++++" % len(self.corpus))
        np.random.shuffle(self.corpus)


class MedicalTestDataset(Dataset):

    def __init__(self, data_path, vocab_path, vocab_size, max_length=100, use_tfidf=False, simple_feature=None):
        super(MedicalTestDataset, self).__init__()
        with open(data_path, 'r') as fin:
            lines = fin.readlines()
        self.corpus = []
        self.use_tfidf = use_tfidf
        self.w2id, self.id2w = load_vocab(vocab_path, vocab_size)
        if self.use_tfidf and simple_feature is None:
            raise ValueError("the simple feature object can not be None when using tf-idf")
        self.simple_feature = simple_feature
        for line in lines:
            report_id, txt = line.strip('\n').split('|,|')
            if use_tfidf:
                txt_ids = txt
            else:
                txt_ids = [self.w2id.get(ids, self.w2id['<unk>']) for ids in txt.split(' ') if len(ids.strip()) > 0]
            self.corpus.append({'report_id': report_id, 'txt': txt_ids})
        self.max_length = max_length

    def __getitem__(self, index):
        item = self.corpus[index]
        report_id = item['report_id']
        report_txt = item['txt']
        if self.use_tfidf:
            report_length = len(report_txt.split())
            report_txt = self.simple_feature.get_tf_idf([report_txt]).reshape(-1)
        else:
            report_length = len(report_txt)
        return report_id, report_txt, report_length

    def __len__(self):
        return len(self.corpus)


class MedicalDataloader(DataLoader):

    def __init__(self, data_path, vocab_path, vocab_size, class_id, num_class, batch_size, shuffle, num_worker,
                 sample, use_tfidf=False,
                 simple_feature=None):
        # data_path, vocab_path, vocab_size, class_id, num_class=17, max_length=100, sample=True,
        #                  use_tfidf=False, simple_feature=None
        dataset = MedicalDataset(data_path, vocab_path, vocab_size, class_id, num_class, sample, use_tfidf=use_tfidf,
                                 simple_feature=simple_feature)

        super(MedicalDataloader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                                num_workers=num_worker,
                                                collate_fn=self.collate_fn)

    def collate_fn(self, data):
        report_ids, report_txt, report_lengths, class_vec, class_label, label = zip(*data)
        if self.dataset.use_tfidf:
            txt_ids = torch.Tensor(report_txt)
        else:
            max_length = max(report_lengths)
            txt_ids = np.zeros((len(report_ids), max_length), dtype=int)
            for i, txt_id in enumerate(report_txt):
                txt_ids[i, :len(txt_id)] = txt_id
            txt_ids = torch.from_numpy(txt_ids)
        class_vec = torch.Tensor(class_vec)
        label = torch.Tensor(label)
        class_label = torch.Tensor(class_label)
        report_lengths = torch.Tensor(report_lengths)
        return report_ids, txt_ids, report_lengths, class_vec, class_label, label

    def re_sample(self, slice_index):
        self.dataset.re_sample(slice_index)


class MedicalTestDataloader(DataLoader):

    def __init__(self, data_path, vocab_path, vocab_size, batch_size, num_worker, use_tfidf=False, simple_feature=None):
        # data_path, vocab_path, vocab_size, max_length=100, use_tfidf=False, simple_feature=None
        dataset = MedicalTestDataset(data_path, vocab_path, vocab_size, use_tfidf=use_tfidf,
                                     simple_feature=simple_feature)
        super(MedicalTestDataloader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=False,
                                                    num_workers=num_worker,
                                                    collate_fn=self.collate_fn)

    def collate_fn(self, data):
        report_ids, report_txt, report_lengths = zip(*data)
        if self.dataset.use_tfidf:
            txt_ids = torch.Tensor(report_txt)
        else:
            max_length = max(report_lengths)
            txt_ids = np.zeros((len(report_ids), max_length), dtype=int)
            for i, txt_id in enumerate(report_txt):
                txt_ids[i, :len(txt_id)] = txt_id
            txt_ids = torch.from_numpy(txt_ids)
        report_lengths = torch.Tensor(report_lengths)
        return report_ids, txt_ids, report_lengths


if __name__ == '__main__':
    # dataloader = MedicalDataloader("/Users/songjixian/Documents/work/tianchi/tc_data/track1_round1_train_20210222.csv",
    #                                17,
    #                                16,
    #                                False,
    #                                1)
    # for report_ids, txt, lengths, class_vec in dataloader:
    #     print(report_ids)
    #     print(txt)
    #     exit(0)
    pass
