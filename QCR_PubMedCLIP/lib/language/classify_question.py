# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         classify_question
# Description:  
# Author:       Boliu.Kelvin
# Date:         2020/5/14
#-------------------------------------------------------------------------------


import torch
from dataset import *
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from language.language_model import WordEmbedding,QuestionEmbedding
import argparse
from torch.nn.init import kaiming_uniform_, xavier_uniform_
import torch.nn.functional as F
import utils
from datetime import datetime
import clip
from transformers import AutoConfig, AutoModel


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def linear(in_dim, out_dim, bias=True):

    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()

    return lin


# change the question b*number*hidden -> b*hidden
class QuestionAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.tanh_gate = linear(300 + dim, dim)
        self.sigmoid_gate = linear(300 + dim, dim)
        self.attn = linear(dim, 1)
        self.dim = dim

    def forward(self, context, question):  #b*12*300 b*12*1024/  or b*77*512 if clip

        if len(question.shape) == 2:   # in case of using clip to encode q
            question = question.unsqueeze(1)
            question = question.expand(-1, 77, -1)
        concated = torch.cat([context, question], -1)  # b * 12 * 300 + 1024 / or 512 if clip
        concated = torch.mul(torch.tanh(self.tanh_gate(concated)), torch.sigmoid(self.sigmoid_gate(concated)))  #b*12*1024 / or b*77*512 if clip
        a = self.attn(concated) # #b*12*1  / or b*77*1 if clip
        attn = F.softmax(a.squeeze(), 1) #b*12 / or b*77 if clip

        ques_attn = torch.bmm(attn.unsqueeze(1), question).squeeze() #b*1024 / or b*512 if clip

        return ques_attn


class typeAttention(nn.Module):
    def __init__(self, size_question, path_init):
        super(typeAttention, self).__init__()
        self.w_emb = WordEmbedding(size_question, 300, 0.0, False)
        self.w_emb.init_embedding(path_init)
        self.q_emb = QuestionEmbedding(300, 1024, 1, False, 0.0, 'GRU')
        self.q_final = QuestionAttention(1024)
        self.f_fc1 = linear(1024, 2048)
        self.f_fc2 = linear(2048, 1024)
        self.f_fc3 = linear(1024, 1024)

    def forward(self, question):
        question = question[0]
        w_emb = self.w_emb(question)
        q_emb = self.q_emb.forward_all(w_emb)  # [batch, q_len, q_dim]
        q_final = self.q_final(w_emb, q_emb)  # b, 1024

        x_f = self.f_fc1(q_final)
        x_f = F.relu(x_f)
        x_f = self.f_fc2(x_f)
        x_f = F.dropout(x_f)
        x_f = F.relu(x_f)
        x_f = self.f_fc3(x_f)

        return x_f


class classify_model(nn.Module):
    def __init__(self,size_question,path_init):
        super(classify_model,self).__init__()
        self.w_emb = WordEmbedding(size_question,300, 0.0, False)
        self.w_emb.init_embedding(path_init)
        self.q_emb = QuestionEmbedding(300, 1024 , 1, False, 0.0, 'GRU')
        self.q_final = QuestionAttention(1024)
        self.f_fc1 = linear(1024,256)
        self.f_fc2 = linear(256,64)
        self.f_fc3 = linear(64,2)

    def forward(self,question):
        question = question[0]

        w_emb = self.w_emb(question)
        q_emb = self.q_emb.forward_all(w_emb)  # [batch, q_len, q_dim]
        q_final = self.q_final(w_emb,q_emb) #b, 1024

        x_f = self.f_fc1(q_final)
        x_f = F.relu(x_f)
        x_f = self.f_fc2(x_f)
        x_f = F.dropout(x_f)
        x_f = F.relu(x_f)
        x_f = self.f_fc3(x_f)

        return x_f


def parse_args():
    parser = argparse.ArgumentParser(description="Med VQA over MAC")
    # GPU config
    parser.add_argument('--seed', type=int, default=5
                        , help='random seed for gpu.default:5')
    parser.add_argument('--gpu', type=int, default=0,
                        help='use gpu device. default:0')
    args = parser.parse_args()
    return args


# Evaluation
def evaluate(model, dataloader,logger,device):
    score = 0
    number =0
    model.eval()
    with torch.no_grad():
        for i,row in enumerate(dataloader):
            image_data, question, target, answer_type, question_type, phrase_type, answer_target = row
            question, answer_target = question.to(device), answer_target.to(device)
            output = model(question)
            pred = output.data.max(1)[1]
            correct = pred.eq(answer_target.data).cpu().sum()
            score+=correct.item()
            number+=len(answer_target)

        score = score / number * 100.

    logger.info('[Validate] Val_Acc:{:.6f}%'.format(score))
    return score
