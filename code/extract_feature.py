#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/28 11:20 上午
# @Author  : 宋继贤
# @Description  : 
# @File    : extract_feature.py
# @Software: PyCharm


import os
import torch
import numpy as np
from transformers import BertModel, BertTokenizer


def extract_bert_embedd(bert_path, vocab_name, data_path, feature_path, label_path):
    tokenizer = BertTokenizer.from_pretrained(os.path.join(bert_path, vocab_name))
    bert = BertModel.from_pretrained(bert_path)
    vectors = []
    labels = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            text = line.strip('\n').split('\t')[0]
            label = line.strip('\n').split('\t')[1]
            text_encode = tokenizer.encode(text)
            input_tensor = torch.tensor(text_encode).unsqueeze(0)
            output = bert(input_tensor)
            embedding_of_last = output[0]
            cls_vector = embedding_of_last[:, 0, :]
            vectors.append(cls_vector.squeeze(0).detach().numpy())
            labels.append(label)
    np.save(feature_path, np.array(vectors, dtype=np.float32))
    with open(label_path, 'w', encoding='utf-8') as fout:
        fout.writelines('\n'.join(labels))


if __name__ == '__main__':
    pass