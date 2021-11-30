# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         classify_question
# Description:  
# Author:       Boliu.Kelvin
# Date:         2020/5/14
#-------------------------------------------------------------------------------


import _init_paths
import torch
from config import cfg, update_config
from dataset import *
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from utils.create_dictionary import Dictionary
from language.language_model import WordEmbedding,QuestionEmbedding
import argparse
from torch.nn.init import kaiming_uniform_, xavier_uniform_
import torch.nn.functional as F
from utils import utils
from datetime import datetime
from language.classify_question import classify_model


def parse_args():
    parser = argparse.ArgumentParser(description="Type classifier")
    # GPU config
    parser.add_argument(
            "--cfg",
            help="decide which cfg to use",
            required=False,
            default="/home/test.yaml",
            type=str,
            )
    # GPU config
    parser.add_argument('--gpu', type=int, default=0,
                        help='use gpu device. default:0')
    parser.add_argument('--seed', type=int, default=5
                        , help='random seed for gpu.default:5')
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
            question[0], answer_target = question[0].to(device), answer_target.to(device)
            output = model(question)
            pred = output.data.max(1)[1]
            correct = pred.eq(answer_target.data).cpu().sum()
            score+=correct.item()
            number+=len(answer_target)

        score = score / number * 100.

    logger.info('[Validate] Val_Acc:{:.6f}%'.format(score))
    return score


if __name__=='__main__':
    args = parse_args()
    update_config(cfg, args)
    dataroot = cfg.DATASET.DATA_DIR 
    # # set GPU device
    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")

    # Fixed ramdom seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    d = Dictionary.load_from_file(os.path.join(dataroot, 'dictionary.pkl'))
    train_dataset = VQASLAKEFeatureDataset('train', cfg, d, dataroot=dataroot)
    train_data = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)

    val_dataset = VQASLAKEFeatureDataset('test', cfg, d, dataroot=dataroot)
    val_data = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

    net = classify_model(d.ntoken, os.path.join(dataroot, 'glove6b_init_300d.npy'))
    net =net.to(device)

    run_timestamp = datetime.now().strftime("%Y%b%d-%H%M%S")
    ckpt_path = os.path.join('./log', run_timestamp)
    utils.create_dir(ckpt_path)
    model_path = os.path.join(ckpt_path, "best_model.pth")
    # create logger
    logger = utils.Logger(os.path.join(ckpt_path, 'medVQA.log')).get_logger()
    logger.info(">>>The net is:")
    logger.info(net)
    logger.info(">>>The args is:")
    logger.info(args.__repr__())

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    
    epochs = 200
    best_eval_score = 0
    best_epoch = 0
    for epoch in range(epochs):
        net.train()
        acc = 0.
        number_dataset = 0
        total_loss = 0
        for i, row in enumerate(train_data):
            image_data, question, target, answer_type, question_type, phrase_type, answer_target = row
            question[0], answer_target = question[0].to(device), answer_target.to(device)
            optimizer.zero_grad()
            output = net(question)
            loss = criterion(output,answer_target)
            loss.backward()
            optimizer.step()
            pred = output.data.max(1)[1]
            correct = (pred==answer_target).data.cpu().sum()
            
            acc += correct.item()
            number_dataset += len(answer_target)
            total_loss+= loss
        
        total_loss /= len(train_data)
        acc = acc/ number_dataset * 100.

        logger.info('-------[Epoch]:{}-------'.format(epoch))
        logger.info('[Train] Loss:{:.6f} , Train_Acc:{:.6f}%'.format(total_loss, acc
                                                                     ))
        # Evaluation
        if val_data is not None:
            eval_score = evaluate(net, val_data, logger, device)
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                best_epoch = epoch
                utils.save_model(model_path, net, best_epoch, eval_score)
            logger.info('[Result] The best acc is {:.6f}% at epoch {}'.format(best_eval_score, best_epoch))
