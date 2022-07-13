# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         utils
# Description:  copy from Hengyuan Hu's repository.  https://github.com/hengyuan-hu/bottom-up-attention-vqa
# Date:         2020/4/6
#-------------------------------------------------------------------------------

from __future__ import print_function

import errno
import os
import re
import collections
import numpy as np
import operator
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._six import string_classes
from torch.utils.data.dataloader import default_collate
import logging
from utils.create_dictionary import Dictionary
import itertools
import _pickle as cPickle
import json
import pandas as pd


EPS = 1e-7
numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}
def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)

def assert_array_eq(real, expected):
    assert (np.abs(real-expected) < EPS).all(), \
        '%s (true) vs %s (expected)' % (real, expected)

def weights_init(m):
    """custom weights initialization."""
    cname = m.__class__
    if cname == nn.Linear or cname == nn.Conv2d or cname == nn.ConvTranspose2d:
        m.weight.data.normal_(0.0, 0.02)
    elif cname == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        print('%s is not initialized.' % cname)

class Logger(object):
    def __init__(self, filename, level=logging.INFO,
                 format='%(asctime)s %(levelname)s %(message)s',
                 datefmt='%a, %d %b %Y %H:%M:%S', filemode='w'):
        self.level = level
        self.format = format
        self.datefmt = datefmt
        self.filename = filename
        self.filemode = filemode
        logging.basicConfig(level=self.level,
                            format=self.format,
                            datefmt=self.datefmt,
                            filename=self.filename,
                            filemode=self.filemode)
        self._set_streaming_handler()

    def _set_streaming_handler(self, level=logging.INFO, formatter='%(asctime)s %(levelname)-8s %(message)s'):
        console = logging.StreamHandler()
        console.setLevel(level)
        curr_formatter = logging.Formatter(formatter)
        console.setFormatter(curr_formatter)
        logging.getLogger(self.filename).addHandler(console)

    def get_logger(self):
        return logging.getLogger(self.filename)

def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

def print_model(model, logger):
    print(model)
    nParams = 0
    for w in model.parameters():
        nParams += functools.reduce(operator.mul, w.size(), 1)
    if logger:
        logger.write('nParams=\t'+str(nParams))

def save_model(path, model, epoch, eval_score, open_score=None, close_score=None, optimizer=None):
    model_dict = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'eval_score': eval_score,
            'open_score': open_score,
            'close_score': close_score
        }
    if optimizer is not None:
        model_dict['optimizer_state'] = optimizer.state_dict()

    torch.save(model_dict, path)


def trim_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    _use_shared_memory = True
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
        out = None
        if 1 < batch[0].dim(): # image features
            max_num_boxes = max([x.size(0) for x in batch])
            if _use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = len(batch) * max_num_boxes * batch[0].size(-1)
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            # warning: F.pad returns Variable!
            return torch.stack([F.pad(x, (0,0,0,max_num_boxes-x.size(0))).data for x in batch], 0, out=out)
        else:
            if _use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [trim_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))

def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    # glove_file = glove_file if args.use_TDIUC else os.path.join(args.TDIUC_dir, 'glove', glove_file.split('/')[-1])
    with open(glove_file, 'r', encoding='utf-8') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
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

# --------------------FAIRSEQ functions---------------------------
def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda()
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_cuda(value)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)

def item(tensor):
    if hasattr(tensor, 'item'):
        return tensor.item()
    if hasattr(tensor, '__getitem__'):
        return tensor[0]
    return tensor

def clip_grad_norm_(tensor, max_norm):
    grad_norm = item(torch.norm(tensor))
    if grad_norm > max_norm > 0:
        clip_coef = max_norm / (grad_norm + 1e-6)
        tensor.mul_(clip_coef)
    return grad_norm

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

def get_size_of_largest_vqa_batch(dataloader):
    largest_v = None
    largest_b = None
    largest_q = None
    largest_a = None
    v, b, q, a = iter(dataloader).next()
    # ignore 1st dimension (batch size)
    largest_v = v.size()[1]
    largest_b = b.size()[1]
    largest_q = q.size()[1]
    largest_a = a.size()[1]
    for i, (v, b, q, a) in enumerate(dataloader):
        if largest_v > v.size()[1]:
            pass


def tfidf_loading(use_tfidf, w_emb, cfg):
    data_dir = cfg.DATASET.DATA_DIR
    if use_tfidf:
        if cfg.TRAIN.QUESTION.USEDATA:
            dict = Dictionary.load_from_file(os.path.join(data_dir, 'dictionary.pkl'))

        # load extracted tfidf and weights from file for saving loading time
        if cfg.TRAIN.QUESTION.USEDATA:
            if os.path.isfile(os.path.join(data_dir, 'embed_tfidf_weights.pkl')) == True:
                print("Loading embedding tfidf and weights from file")
                with open(os.path.join(data_dir ,'embed_tfidf_weights.pkl'), 'rb') as f:
                    w_emb = torch.load(f)
                print("Load embedding tfidf and weights from file successfully")
            else:
                print("Embedding tfidf and weights haven't been saving before")
                # tfidf, weights = tfidf_from_questions(['train'], args, dict)
                tfidf, weights = tfidf_from_questions(['train', 'test'], cfg, dict)
                w_emb.init_embedding(os.path.join(data_dir, 'glove6b_init_300d.npy'), tfidf, weights)
                with open(os.path.join(data_dir ,'embed_tfidf_weights.pkl'), 'wb') as f:
                    torch.save(w_emb, f)
                print("Saving embedding with tfidf and weights successfully")
    return w_emb


def tfidf_from_questions(names, cfg, dictionary):
    inds = [[], []] # rows, cols for uncoalesce sparse matrix
    df = dict()
    N = len(dictionary)
    target = cfg.DATASET.DATASET.lower()
    if cfg.TRAIN.QUESTION.USEDATA:
        dataroot = cfg.DATASET.DATA_DIR
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
            else:
                print(c[1])

    if 'rad' in target:
        for name in names:
            assert name in ['train', 'test']
            question_path = os.path.join(dataroot, name + 'set.json')
            questions = json.load(open(question_path))
            for question in questions:
                populate(inds, df, question['question'])
    elif 'slake' in target:
        for name in names:
            assert name in ['train', 'test']
            question_path = os.path.join(dataroot, name + '.json')
            questions = json.load(open(question_path))
            for question in questions:
                if question['q_lang'] == "en":
                    populate(inds, df, question['question'])
                else:
                    continue
    else:
        raise ValueError(f"Target {target} not supported!")

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
    glove_file = os.path.join('./data', 'glove.6B', 'glove.6B.%dd.txt' % emb_dim)
    weights, word2emb = create_glove_embedding_init(dictionary.idx2word[N:], glove_file)
    print('tf-idf stochastic matrix (%d x %d) is generated.' % (tfidf.size(0), tfidf.size(1)))

    return tfidf, weights
