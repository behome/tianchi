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
import pickle
from transformers import BertModel, BertTokenizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


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


class SimpleFeature:

    def __init__(self, resume=False, vocab_path=None, tfidf_path=None):
        if resume and (vocab_path is None or tfidf_path is None):
            raise ValueError("The vocabulary path is None")
        self.resume = resume
        if resume:
            self.vectorizer = CountVectorizer(vocabulary=pickle.load(open(vocab_path, 'rb')))
            self.tfidf = pickle.load(open(tfidf_path, 'rb'))
        else:
            self.vectorizer = CountVectorizer()
            self.tfidf = TfidfTransformer()

    def get_train_feature(self, corpus, save=True, vocab_path=None, tfidf_path=None):
        if save and vocab_path is None:
            raise ValueError("The vocabulary path is None")
        vec = self.vectorizer.fit_transform(corpus)
        with open(vocab_path, 'wb') as fw:
            pickle.dump(self.vectorizer.vocabulary_, fw)
        tfidf = self.tfidf.fit_transform(vec)
        with open(tfidf_path, 'wb') as fw:
            pickle.dump(self.tfidf, fw)
        return vec.toarray(), tfidf.toarray()

    def get_tf_idf(self, content):
        bow = self.vectorizer.transform(content)
        return self.tfidf.transform(bow).toarray()

    def get_bow(self, content):
        return self.vectorizer.transform(content).toarray()


if __name__ == '__main__':
    with open("../tc_data/track1_round1_train_20210222.csv", 'r') as f:
        lines = f.readlines()
    corpus = []
    for line in lines:
        report_id, txt, classes = line.strip('\n').split('|,|')
        corpus.append(txt)
    feature = SimpleFeature(vocab_path='../user_data/model_data/vocab.pkl', tfidf_path='../user_data/model_data/tf_idf.pkl')
    feature.get_train_feature(corpus, save=False, vocab_path='../user_data/model_data/vocab.pkl', tfidf_path='../user_data/model_data/tf_idf.pkl')

    print(feature.vectorizer.vocabulary_)
    for key, value in feature.vectorizer.vocabulary_.items():
        print(key, value)
    # txt = '623 328 399 698 493 338 266 14 177 415 511 647 693 852 60 328 380 172 54 788 591 487'
    # vec = feature.get_tf_idf([txt])
    # print(vec.shape)
