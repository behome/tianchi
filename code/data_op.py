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


class MedicalDataset(Dataset):

    def __init__(self, data_path, num_class=17, max_length=100):
        super(MedicalDataset, self).__init__()
        with open(data_path, 'r') as fin:
            lines = fin.readlines()
        self.corpus = []
        for line in lines:
            report_id, txt, classes = line.strip('\n').split('|,|')
            txt_ids = [int(ids) for ids in txt.split(' ') if len(ids.strip()) > 0]
            classes = [int(class_id) for class_id in classes.split(' ') if len(class_id.strip()) > 0]
            self.corpus.append({'report_id': report_id, 'txt': txt_ids, 'classes': classes})
        self.num_class = num_class
        self.max_length = max_length

    def __getitem__(self, index):
        item = self.corpus[index]
        report_id = item['report_id']
        report_txt = item['txt']
        report_length = len(report_txt)
        classes = item['classes']
        class_vec = np.zeros((17,))
        for class_id in classes:
            class_vec[class_id] = 1.0
        return report_id, report_txt, report_length, class_vec

    def __len__(self):
        return len(self.corpus)


class MedicalTestDataset(Dataset):

    def __init__(self, data_path, max_length=100):
        super(MedicalTestDataset, self).__init__()
        with open(data_path, 'r') as fin:
            lines = fin.readlines()
        self.corpus = []
        for line in lines:
            report_id, txt = line.strip('\n').split('|,|')
            txt_ids = [int(ids) for ids in txt.split(' ') if len(ids.strip()) > 0]
            self.corpus.append({'report_id': report_id, 'txt': txt_ids})
        self.max_length = max_length

    def __getitem__(self, index):
        item = self.corpus[index]
        report_id = item['report_id']
        report_txt = item['txt']
        report_length = len(report_txt)
        return report_id, report_txt, report_length

    def __len__(self):
        return len(self.corpus)


class MedicalDataloader(DataLoader):

    def __init__(self, data_path, num_class, batch_size, shuffle, num_worker):
        dataset = MedicalDataset(data_path, num_class)

        super(MedicalDataloader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                                num_workers=num_worker,
                                                collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(data):
        report_ids, report_txt, report_lengths, class_vec = zip(*data)
        max_length = max(report_lengths)
        txt_ids = np.zeros((len(report_ids), max_length), dtype=int)
        for i, txt_id in enumerate(report_txt):
            txt_ids[i, :len(txt_id)] = txt_id
        class_vec = torch.Tensor(class_vec)
        txt_ids = torch.from_numpy(txt_ids)
        report_lengths = torch.Tensor(report_lengths)
        return report_ids, txt_ids, report_lengths, class_vec


class MedicalTestDataloader(DataLoader):

    def __init__(self, data_path, batch_size, num_worker):
        dataset = MedicalTestDataset(data_path)

        super(MedicalTestDataloader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=False,
                                                    num_workers=num_worker,
                                                    collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(data):
        report_ids, report_txt, report_lengths = zip(*data)
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
