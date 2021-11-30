# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         model
# Description:  BAN model [Bilinear attention + Bilinear residual network]
#-------------------------------------------------------------------------------
import torch
import torch.nn as nn
from language.language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from network.connect import FCNet
from network.connect import BCNet
from network.counting import Counter
from utils.utils import tfidf_loading
from network.maml import SimpleCNN
from network.auto_encoder import Auto_Encoder_Model
from torch.nn.utils.weight_norm import weight_norm
from language.classify_question import typeAttention 
import clip
import os


# Bilinear Attention
class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse, dropout=[.2,.5]):  #128, 1024, 1024,2
        super(BiAttention, self).__init__()

        self.glimpse = glimpse
        self.logits = weight_norm(BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3),
            name='h_mat', dim=None)

    def forward(self, v, q, v_mask=True):  # v:32,1,128; q:32,12,1024
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v, q)  # b x g x v x q

        if v_mask:
            mask = (0 == v.abs().sum(2)).unsqueeze(1).unsqueeze(3).expand(logits.size())
            logits.data.masked_fill_(mask.data, -float('inf'))

        p = nn.functional.softmax(logits.view(-1, self.glimpse, v_num * q_num), 2)
        return p.view(-1, self.glimpse, v_num, q_num), logits


class BiResNet(nn.Module):
    def __init__(self, cfg, dataset, priotize_using_counter=False):
        super(BiResNet,self).__init__()
        # Optional module: counter
        use_counter = cfg.TRAIN.ATTENTION.USE_COUNTER if priotize_using_counter is None else priotize_using_counter
        if use_counter or priotize_using_counter:
            objects = 10  # minimum number of boxes
        if use_counter or priotize_using_counter:
            counter = Counter(objects)
        else:
            counter = None
        # # init Bilinear residual network
        b_net = []   # bilinear connect :  (XTU)T A (YTV)
        q_prj = []   # output of bilinear connect + original question-> new question    Wq_ +q
        c_prj = []
        for i in range(cfg.TRAIN.ATTENTION.GLIMPSE):
            b_net.append(BCNet(dataset.v_dim, cfg.TRAIN.QUESTION.HID_DIM, cfg.TRAIN.QUESTION.HID_DIM, None, k=1))
            q_prj.append(FCNet([cfg.TRAIN.QUESTION.HID_DIM, cfg.TRAIN.QUESTION.HID_DIM], '', .2))
            if use_counter or priotize_using_counter:
                c_prj.append(FCNet([objects + 1, cfg.TRAIN.QUESTION.HID_DIM], 'ReLU', .0))

        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.c_prj = nn.ModuleList(c_prj)
        self.cfg = cfg

    def forward(self, v_emb, q_emb,att_p):
        b_emb = [0] * self.cfg.TRAIN.ATTENTION.GLIMPSE
        for g in range(self.cfg.TRAIN.ATTENTION.GLIMPSE):
            b_emb[g] = self.b_net[g].forward_with_weights(v_emb, q_emb, att_p[:,g,:,:]) # b x l x h
            # atten, _ = logits[:,g,:,:].max(2)
            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
        return q_emb.sum(1)


def seperate(v,q,a,att, answer_target, n_unique_close):     #q: b x 12 x 1024  v:  b x 1 x 128 answer_target : 1 x b
    indexs_open = []
    indexs_close = []

    for i in range(len(answer_target)):
        if answer_target[i]==0:
            indexs_close.append(i)
        else:
            indexs_open.append(i)
        if len(q.shape) == 2:  # in case of using clip to encode q
            q = q.unsqueeze(1)
    return v[indexs_open,:,:],v[indexs_close,:,:],q[indexs_open,:,:],\
            q[indexs_close,:,:],a[indexs_open, n_unique_close:],a[indexs_close,:n_unique_close],att[indexs_open,:],att[indexs_close,:], indexs_open, indexs_close
 

# Create BAN model
class BAN_Model(nn.Module):
    def __init__(self, dataset, cfg, device):
        super(BAN_Model, self).__init__()

        self.cfg = cfg
        self.dataset = dataset
        self.device = device
        # init word embedding module, question embedding module, biAttention network, bi_residual network, and classifier
        self.w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, cfg.TRAIN.QUESTION.CAT)
        self.q_emb = QuestionEmbedding(600 if cfg.TRAIN.QUESTION.CAT else 300, cfg.TRAIN.QUESTION.HID_DIM, 1, False, .0, cfg.TRAIN.QUESTION.RNN)

        # for close att+ resnet + classify
        self.close_att = BiAttention(dataset.v_dim, cfg.TRAIN.QUESTION.HID_DIM, cfg.TRAIN.QUESTION.HID_DIM, cfg.TRAIN.ATTENTION.GLIMPSE)
        self.close_resnet = BiResNet(cfg, dataset)
        self.close_classifier = SimpleClassifier(cfg.TRAIN.QUESTION.CLS_HID_DIM, cfg.TRAIN.QUESTION.CLS_HID_DIM * 2, dataset.num_close_candidates, cfg)

        # for open_att + resnet + classify
        self.open_att = BiAttention(dataset.v_dim, cfg.TRAIN.QUESTION.HID_DIM, cfg.TRAIN.QUESTION.HID_DIM, cfg.TRAIN.ATTENTION.GLIMPSE)
        self.open_resnet = BiResNet(cfg, dataset)
        self.open_classifier = SimpleClassifier(cfg.TRAIN.QUESTION.CLS_HID_DIM, cfg.TRAIN.QUESTION.CLS_HID_DIM * 2, dataset.num_open_candidates, cfg)

        self.bbn_classifier = SimpleClassifier(cfg.TRAIN.QUESTION.CLS_HID_DIM*2, cfg.TRAIN.QUESTION.CLS_HID_DIM * 3, dataset.num_ans_candidates, cfg)
        path = os.path.join(self.cfg.DATASET.DATA_DIR, "glove6b_init_300d.npy")
        self.typeatt = typeAttention(dataset.dictionary.ntoken, path)
        
        # build and load pre-trained MAML model
        if cfg.TRAIN.VISION.MAML:
            weight_path = cfg.DATASET.DATA_DIR + '/' + cfg.TRAIN.VISION.MAML_PATH
            print('load initial weights MAML from: %s' % (weight_path))
            self.maml = SimpleCNN(weight_path, cfg.TRAIN.OPTIMIZER.EPS_CNN, cfg.TRAIN.OPTIMIZER.MOMENTUM_CNN)
        # build and load pre-trained Auto-encoder model
        if cfg.TRAIN.VISION.AUTOENCODER:
            self.ae = Auto_Encoder_Model()
            weight_path = cfg.DATASET.DATA_DIR + '/' + cfg.TRAIN.VISION.AE_PATH
            print('load initial weights DAE from: %s' % (weight_path))
            self.ae.load_state_dict(torch.load(weight_path))
            self.convert = nn.Linear(16384, 64)
        # build and load pre-trained CLIP model
        if cfg.TRAIN.VISION.CLIP:
            self.clip, _ = clip.load(cfg.TRAIN.VISION.CLIP_VISION_ENCODER, jit=False)
            if not cfg.TRAIN.VISION.CLIP_ORG:
                checkpoint = torch.load(cfg.TRAIN.VISION.CLIP_PATH)
                self.clip.load_state_dict(checkpoint['state_dict'])
            self.clip = self.clip.float()
        # Loading tfidf weighted embedding
        if cfg.TRAIN.QUESTION.TFIDF:
            self.w_emb = tfidf_loading(cfg.TRAIN.QUESTION.TFIDF, self.w_emb, cfg)
        # Loading the other net
        if cfg.TRAIN.VISION.OTHER_MODEL:
            pass
        
    def forward(self, v, q, a, answer_target):
        """Forward
        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [[batch_size, seq_length], [batch_size, seq_length]]
        return: logits, not probs
        """
        # get visual feature
        if self.cfg.TRAIN.VISION.MAML:
            maml_v_emb = self.maml(v[0]).unsqueeze(1)
            v_emb = maml_v_emb
        if self.cfg.TRAIN.VISION.AUTOENCODER:
            encoder = self.ae.forward_pass(v[1])
            decoder = self.ae.reconstruct_pass(encoder)
            ae_v_emb = encoder.view(encoder.shape[0], -1)
            ae_v_emb = self.convert(ae_v_emb).unsqueeze(1)
            v_emb = ae_v_emb
        if self.cfg.TRAIN.VISION.CLIP:
            clip_v_emb = self.clip.encode_image(v[2]).unsqueeze(1)
            v_emb = clip_v_emb
        if self.cfg.TRAIN.VISION.MAML and self.cfg.TRAIN.VISION.AUTOENCODER:
            v_emb = torch.cat((maml_v_emb, ae_v_emb), 2)
        if self.cfg.TRAIN.VISION.CLIP and self.cfg.TRAIN.VISION.AUTOENCODER:
            v_emb = torch.cat((clip_v_emb, ae_v_emb), 2)
        if self.cfg.TRAIN.VISION.MAML and self.cfg.TRAIN.VISION.AUTOENCODER and self.cfg.TRAIN.VISION.CLIP:
            v_emb = torch.cat((maml_v_emb, ae_v_emb, clip_v_emb), 2)
        if self.cfg.TRAIN.VISION.OTHER_MODEL:
            pass

        # get type attention
        type_att = self.typeatt(q)
        # get lextual feature    global 
        w_emb = self.w_emb(q[0])
        q_emb = self.q_emb.forward_all(w_emb) # [batch, q_len, q_dim]

        # get open & close feature
        v_open, v_close, q_open, q_close,a_open, a_close, typeatt_open, typeatt_close, _, _ = seperate(v_emb,q_emb,a,type_att, answer_target, self.dataset.num_close_candidates)

        # diverse Attention -> (open + close)
        att_close, _ = self.close_att(v_close,q_close)
        att_open, _ = self.open_att(v_open,q_open)

        # bilinear residual network
        last_output_close = self.close_resnet(v_close,q_close,att_close)
        last_output_open = self.open_resnet(v_open,q_open,att_open)

        #type attention (5.19 try)
        last_output_close = last_output_close * typeatt_close
        last_output_open = last_output_open * typeatt_open
        #qtype attention (5.19 try)

        if self.cfg.TRAIN.VISION.AUTOENCODER:
                return last_output_close,last_output_open,a_close,a_open, decoder
        return last_output_close,last_output_open,a_close, a_open

    def classify(self, close_feat, open_feat):
        return self.close_classifier(close_feat), self.open_classifier(open_feat)
    def bbn_classify(self, bbn_mixed_feature):
        return self.bbn_classifier(bbn_mixed_feature)
    
    def forward_classify(self,v,q,a,classify, n_unique_close):
        # get visual feature
        if self.cfg.TRAIN.VISION.MAML:
            maml_v_emb = self.maml(v[0]).unsqueeze(1)
            v_emb = maml_v_emb
        if self.cfg.TRAIN.VISION.AUTOENCODER:
            encoder = self.ae.forward_pass(v[1])
            decoder = self.ae.reconstruct_pass(encoder)
            ae_v_emb = encoder.view(encoder.shape[0], -1)
            ae_v_emb = self.convert(ae_v_emb).unsqueeze(1)
            v_emb = ae_v_emb
        if self.cfg.TRAIN.VISION.CLIP:
            clip_v_emb = self.clip.encode_image(v[2]).unsqueeze(1)
            v_emb = clip_v_emb
        if self.cfg.TRAIN.VISION.MAML and self.cfg.TRAIN.VISION.AUTOENCODER:
            v_emb = torch.cat((maml_v_emb, ae_v_emb), 2)
        if self.cfg.TRAIN.VISION.CLIP and self.cfg.TRAIN.VISION.AUTOENCODER:
            v_emb = torch.cat((clip_v_emb, ae_v_emb), 2)
        if self.cfg.TRAIN.VISION.MAML and self.cfg.TRAIN.VISION.AUTOENCODER and self.cfg.TRAIN.VISION.CLIP:
            v_emb = torch.cat((maml_v_emb, ae_v_emb, clip_v_emb), 2)
        if self.cfg.TRAIN.VISION.OTHER_MODEL:
            pass

        # get type attention
        type_att = self.typeatt(q)
        # get lextual feature    global 
        w_emb = self.w_emb(q[0])
        q_emb = self.q_emb.forward_all(w_emb) # [batch, q_len, q_dim]
        # get open & close feature
        answer_target = classify(q)
        _,predicted=torch.max(answer_target,1)
        v_open, v_close, q_open, q_close,a_open, a_close,typeatt_open, typeatt_close, indexs_open, indexs_close  = seperate(v_emb,q_emb,a,type_att, predicted, self.dataset.num_close_candidates)

        # diverse Attention -> (open + close)
        att_close, _ = self.close_att(v_close,q_close)
        att_open, _ = self.open_att(v_open,q_open)

        # bilinear residual network
        last_output_close = self.close_resnet(v_close,q_close,att_close)
        last_output_open = self.open_resnet(v_open,q_open,att_open)

        # type attention (5.19 try)
        last_output_close = last_output_close * typeatt_close
        last_output_open = last_output_open * typeatt_open
        if self.cfg.TRAIN.VISION.AUTOENCODER:
                return last_output_close,last_output_open,a_close,a_open, decoder, indexs_open, indexs_close
        return last_output_close,last_output_open,a_close, a_open, indexs_open, indexs_close
