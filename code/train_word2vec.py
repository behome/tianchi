""" pretrain a word2vec on the corpus"""
import argparse
import os
from os.path import join, exists
from time import time
from datetime import timedelta

import gensim


class Sentences(object):
    """ needed for gensim word2vec training"""
    def __init__(self, data_path):
        with open(data_path, 'r') as fin:
            self.lines = fin.readlines()

    def __iter__(self):
        for line in self.lines:
            report_id, txt, classes = line.strip('\n').split('|,|')
            yield txt.lower().split()


def main(args):
    start = time()
    save_dir = args.path
    if not exists(save_dir):
        os.makedirs(save_dir)
    sentences = Sentences(args.data_path)
    model = gensim.models.Word2Vec(
        size=args.dim, min_count=5, workers=16, sg=1)
    model.build_vocab(sentences)
    print('vocab built in {}'.format(timedelta(seconds=time()-start)))
    model.train(sentences,
                total_examples=model.corpus_count, epochs=model.iter)

    model.save(join(save_dir, 'word2vec.{}d.{}.bin'.format(
        args.dim, len(model.wv.vocab))))
    model.wv.save_word2vec_format(join(
        save_dir,
        'word2vec.{}d.{}.w2v'.format(args.dim, len(model.wv.vocab))
    ))

    print('word2vec trained in {}'.format(timedelta(seconds=time()-start)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='train word2vec embedding used for model initialization'
    )
    parser.add_argument("--data_path", type=str, default='./tc_data/track1_round1_train_20210222.csv')
    parser.add_argument('--path', type=str, default='./user_data/model_data/', help='root of the model')
    parser.add_argument('--dim', action='store', type=int, default=256)
    args = parser.parse_args()

    main(args)
