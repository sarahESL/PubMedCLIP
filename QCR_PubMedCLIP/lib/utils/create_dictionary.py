# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         create_dictionary
# Description:  question->word->dictionary for validation & training
# Author:       Boliu.Kelvin
# Date:         2020/4/5
#-------------------------------------------------------------------------------

import argparse
import json
import os
import pandas as pd
import numpy as np
import _pickle as cPickle
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        if "? -yes/no" in sentence:
            sentence = sentence.replace("? -yes/no", "")
        if "? -open" in sentence:
            sentence = sentence.replace("? -open", "")
        if "? - open" in sentence:
            sentence = sentence.replace("? - open", "")
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('...', '').replace('x ray', 'x-ray').replace('.', '')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                # if a word is not in dictionary, it will be replaced with the last word of dictionary.
                tokens.append(self.word2idx.get(w, self.padding_idx-1))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def create_dictionary(dataroot, dataset_name, train_file, test_file):
    dictionary = Dictionary()
    questions = []
    files = [train_file, test_file]
    for path in files:
        data_path = os.path.join(data, path)
        with open(data_path) as f:
            d = json.load(f)
        df = pd.DataFrame(d)
        if dataset_name.lower() in ["slake", "vqa-slake", "vqa_slake"]:
            df = df[df['q_lang']=="en"]
        print("processing the {}".format(path))
        for id, row in df.iterrows():
            dictionary.tokenize(row['question'], True)     #row[0]: id , row[1]: question , row[2]: answer

    return dictionary

def create_glove_embedding_init(idx2word, glove_file):
    print('creating glove embeddings...')
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Med VQA")
    parser.add_argument("inputpath", type=str, help="Path to input data")
    parser.add_argument("--dataset", type=str, help="Name of the dataset", default="slake")
    parser.add_argument("--trainfile", type=str, help="Name of the train file", default="train.json")
    parser.add_argument("--testfile", type=str, help="Name of the test file", default="test.json")
    args = parser.parse_args()
    data = args.inputpath
    dataset = args.dataset
    train_file = args.trainfile
    test_file = args.testfile
    d = create_dictionary(data, dataset, train_file, test_file)
    d.dump_to_file(data + '/dictionary.pkl')

    d = Dictionary.load_from_file(data + '/dictionary.pkl')
    emb_dim = 300
    glove_path = data[:data.rindex('/')]
    glove_file = glove_path + '/glove.6B/glove.6B.%dd.txt' % emb_dim
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)
    np.save(data + '/glove6b_init_%dd.npy' % emb_dim, weights)
    print("Process finished successfully!")
