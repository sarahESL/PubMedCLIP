#!/usr/bin/env python
# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         main
# Description:  the entrance of procedure
# Author:       Sedigheh Eslami 
# Date:         2021/08/06
#-------------------------------------------------------------------------------

import _init_paths
from config import cfg, update_config
import argparse
from dataset import *
import os
from torch.utils.data import DataLoader
import torch
from train import train


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CLIP with a medical dataset.")
    # cfg
    parser.add_argument(
            "--cfg",
            help="decide which cfg to use",
            required=False,
            default="/home/test.yaml",
            type=str,
            )
    # GPU config
    parser.add_argument('--seed', type=int, default=5
                        , help='random seed for gpu.default:5')
    parser.add_argument('--gpu', type=int, default=0,
                        help='use gpu device. default:0')


    args = parser.parse_args()
    return args


if __name__ == '__main__':

    data = cfg.DATASET.DATA_DIR
    args = parse_args()
    args.data_dir = data
    # set GPU device
    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    update_config(cfg, args)
    # Fixed random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    # prepare the dataloader
    train_dataset = ImageTextDataset("train", cfg)
    train_loader = DataLoader(train_dataset, cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
    val_dataset = ImageTextDataset("val", cfg)
    val_loader = DataLoader(val_dataset, cfg.TEST.BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)

    # training phase
    train(cfg, train_loader, val_loader, device)

