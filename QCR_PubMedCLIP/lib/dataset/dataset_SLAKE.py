# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         vqa_dataset
# Description:  
# Author:       Boliu.Kelvin, Sedigheh Eslami
# Date:         2020/5/1
#-------------------------------------------------------------------------------


"""
This code is modified based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa) by Xuan B. Nguyen
"""
from __future__ import print_function
import os
import json
import _pickle as cPickle
import numpy as np
from utils import utils
import torch
from language.language_model import WordEmbedding
from torch.utils.data import Dataset,DataLoader
import itertools
import warnings
import h5py
from PIL import Image
import argparse


with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
COUNTING_ONLY = False

# Following Trott et al. (ICLR 2018)
#   Interpretable Counting for Visual Question Answering
def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
       ('number of' in q.lower() and 'number of the' not in q.lower()) or \
       'amount of' in q.lower() or \
       'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False

def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False

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

def _create_entry(img, data, answer):
    if None!=answer:
        answer.pop('img_name')
        answer.pop('qid')
    entry = {
        'qid' : data['qid'],
        'image_name'    : data['img_name'],
        'image'       : img,
        'question'    : data['question'],
        'answer'      : answer,
        'answer_text': data['answer'],
        'answer_type' : data['answer_type'],
        'question_type': data['content_type']
        }
    return entry

def is_json(myjson):
  try:
    json_object = json.loads(myjson)
  except ValueError:
    return False
  return True

def _load_dataset(dataroot, name, img_id2val, label2ans):
    """Load entries

    img2id: dict {img -> id} id can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test'
    """
    data_path = os.path.join(dataroot, name + '.json')
    samples_all = json.load(open(data_path))
    samples = [sample for sample in samples_all if sample['q_lang']=="en"]
    samples = sorted(samples, key=lambda x: x['qid'])

    answer_path = os.path.join(dataroot, 'cache', '%s_openclose_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['qid'])

    utils.assert_eq(len(samples), len(answers))
    entries = []
    for sample, answer in zip(samples, answers):
        utils.assert_eq(sample['qid'], answer['qid'])
        utils.assert_eq(sample['img_name'], answer['img_name'])
        img_id = sample['img_name']
        if not COUNTING_ONLY or is_howmany(sample['question'], answer, label2ans):
            entries.append(_create_entry(img_id2val[img_id], sample, answer))

    return entries


class VQASLAKEFeatureDataset(Dataset):
    def __init__(self, name, cfg, dictionary, dataroot='data'):
        super(VQASLAKEFeatureDataset, self).__init__()
        question_len = cfg.TRAIN.QUESTION.LENGTH
        self.cfg = cfg
        self.name = name
        assert name in ['train', 'test']
        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))

        # close & open
        self.label2close = cPickle.load(open(os.path.join(dataroot,'cache','close_label2ans.pkl'),'rb'))
        self.label2open = cPickle.load(open(os.path.join(dataroot, 'cache', 'open_label2ans.pkl'), 'rb'))
        self.num_open_candidates = len(self.label2open)
        self.num_close_candidates = len(self.label2close)
        self.num_ans_candidates = self.num_open_candidates + self.num_close_candidates

        # End get the number of answer type class
        self.dictionary = dictionary

        # TODO: load img_id2idx
        self.img_id2idx = json.load(open(os.path.join(dataroot, 'imgid2idx.json')))

        self.entries = _load_dataset(dataroot, name, self.img_id2idx, self.label2ans)
        
         # load image data for MAML module
        if self.cfg.TRAIN.VISION.MAML:
            # TODO: load images
            images_path = os.path.join(dataroot, 'images84x84.pkl')
            print('loading MAML image data from file: '+ images_path)
            self.maml_images_data = cPickle.load(open(images_path, 'rb'))
        # load image data for Auto-encoder module
        if self.cfg.TRAIN.VISION.AUTOENCODER:
            # TODO: load images
            images_path = os.path.join(dataroot, 'images128x128.pkl')
            print('loading DAE image data from file: '+ images_path)
            self.ae_images_data = cPickle.load(open(images_path, 'rb'))
        if self.cfg.TRAIN.VISION.CLIP:
            if self.cfg.TRAIN.VISION.CLIP_VISION_ENCODER == "RN50x4":
                images_path = os.path.join(dataroot, 'images288x288.pkl')
            else:
                images_path = os.path.join(dataroot, 'images250x250.pkl')
            print(f"loading CLIP image data from file: {images_path}")
            self.clip_images_data = cPickle.load(open(images_path, 'rb'))
        
        # tokenization
        self.tokenize(question_len)
        self.tensorize()
        if cfg.TRAIN.VISION.AUTOENCODER and cfg.TRAIN.VISION.MAML:
            self.v_dim = cfg.TRAIN.VISION.V_DIM * 2
        else:
            self.v_dim = cfg.TRAIN.VISION.V_DIM  # see the V_DIM defined in config fiels

    def tokenize(self, max_length):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        
        for entry in self.entries:
            if self.cfg.TRAIN.QUESTION.CLIP:
                clip_tokens = clip.tokenize(entry['question'], context_length=max_length)
                clip_tokens = clip_tokens.tolist()[0]
                if len(clip_tokens) < max_length:
                    # Note here we pad in front of the sentence
                    padding = [self.dictionary.padding_idx] * (max_length - len(clip_tokens))
                    clip_tokens = clip_tokens + padding
                    entry['clip_q_token'] = clip_tokens
                utils.assert_eq(len(clip_tokens), max_length)
            
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        if self.cfg.TRAIN.VISION.MAML:
            self.maml_images_data = torch.from_numpy(self.maml_images_data)
            self.maml_images_data = self.maml_images_data.type('torch.FloatTensor')
        if self.cfg.TRAIN.VISION.AUTOENCODER:
            self.ae_images_data = torch.from_numpy(self.ae_images_data)
            self.ae_images_data = self.ae_images_data.type('torch.FloatTensor')
        if self.cfg.TRAIN.VISION.CLIP:
            self.clip_images_data = torch.from_numpy(self.clip_images_data)
            self.clip_images_data = self.clip_images_data.type('torch.FloatTensor')
        for entry in self.entries:
            question = np.array(entry['q_token'])
            entry['q_token'] = question
            if self.cfg.TRAIN.QUESTION.CLIP:
                clip_question = np.array(entry['clip_q_token'])
                entry['clip_q_token'] = clip_question

            answer = entry['answer']
            if None!=answer:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        question_data = [0, 0]
        answer = entry['answer']
        type = answer['type']
        answer_type = entry['answer_type']
        question_type = entry['question_type']
        phrase_type = "UNDEFINED" 
        image_data = [0, 0, 0]
        if self.cfg.TRAIN.VISION.MAML:
            maml_images_data = self.maml_images_data[entry['image']].reshape(84*84)
            image_data[0] = maml_images_data
        if self.cfg.TRAIN.VISION.AUTOENCODER:
            ae_images_data = self.ae_images_data[entry['image']].reshape(128*128)
            image_data[1] = ae_images_data
        if self.cfg.TRAIN.VISION.CLIP:
            if self.cfg.TRAIN.VISION.CLIP_VISION_ENCODER == "RN50x4":
                clip_images_data = self.clip_images_data[entry['image']].reshape(3*288*288)
            else:
                clip_images_data = self.clip_images_data[entry['image']].reshape(3*250*250)
            image_data[2] = clip_images_data

        question_data[0] = entry['q_token']
        if self.cfg.TRAIN.QUESTION.CLIP:
            question_data[1] = entry['clip_q_token']

        if answer_type == 'CLOSED':
            answer_target = 0
        else :
            answer_target = 1

        if None!=answer:
            labels = answer['labels']
            scores = answer['scores']
            composed_target = torch.zeros(self.num_ans_candidates) # close + open
            if answer_target == 0:
                target = torch.zeros(self.num_close_candidates)
                if labels is not None:
                    target.scatter_(0, labels, scores)
                composed_target[:self.num_close_candidates] = target
            else:
                target = torch.zeros(self.num_open_candidates)
                if labels is not None:
                    target.scatter_(0, labels-self.num_close_candidates, scores)
                composed_target[self.num_close_candidates : self.num_ans_candidates] = target
            if self.name == "test":
                return  image_data,question_data, composed_target, answer_type, question_type, phrase_type, answer_target, entry['image_name'], entry['question'], entry['answer_text']
            else:
                return  image_data,question_data, composed_target, answer_type, question_type, phrase_type, answer_target
        else:
            if self.name == "test":
                return image_data, question_data, answer_type, question_type, phrase_type, answer_target, entry['image_name'], entry['question'], entry['answer_text']
            else:
                return image_data, question_data, answer_type, question_type, phrase_type, answer_target

    def __len__(self):
        return len(self.entries)

def tfidf_from_questions(names, args, dictionary, dataroot='data', target=['rad']):
    inds = [[], []] # rows, cols for uncoalesce sparse matrix
    df = dict()
    N = len(dictionary)
    if args.use_RAD:
        dataroot = args.RAD_dir
    def populate(inds, df, text):
        tokens = dictionary.tokenize(text, True)
        for t in tokens:
            df[t] = df.get(t, 0) + 1
        combin = list(itertools.combinations(tokens, 2))
        for c in combin:
            if c[0] < N:
                inds[0].append(c[0]); inds[1].append(c[1])
            if c[1] < N:
                inds[0].append(c[1]); inds[1].append(c[0])

    if 'rad' in target:
        for name in names:
            assert name in ['train', 'test']
            question_path = os.path.join(dataroot, name + 'set.json')
            questions = json.load(open(question_path))
            for question in questions:
                populate(inds, df, question['question'])

    # TF-IDF
    vals = [1] * len(inds[1])
    for idx, col in enumerate(inds[1]):
        assert df[col] >= 1, 'document frequency should be greater than zero!'
        vals[col] /= df[col]

    # Make stochastic matrix
    def normalize(inds, vals):
        z = dict()
        for row, val in zip(inds[0], vals):
            z[row] = z.get(row, 0) + val
        for idx, row in enumerate(inds[0]):
            vals[idx] /= z[row]
        return vals

    vals = normalize(inds, vals)

    tfidf = torch.sparse.FloatTensor(torch.LongTensor(inds), torch.FloatTensor(vals))
    tfidf = tfidf.coalesce()

    # Latent word embeddings
    emb_dim = 300
    glove_file = os.path.join(dataroot, 'glove', 'glove.6B.%dd.txt' % emb_dim)
    weights, word2emb = utils.create_glove_embedding_init(dictionary.idx2word[N:], glove_file)
    print('tf-idf stochastic matrix (%d x %d) is generated.' % (tfidf.size(0), tfidf.size(1)))

    return tfidf, weights
