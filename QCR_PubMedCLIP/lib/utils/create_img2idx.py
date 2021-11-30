#!/usr/bin/env python


import argparse
import pandas as pd
import os
import json
import numpy as np
import pickle
from tqdm import tqdm


def create_img2idx(train_json_path, val_json_path, out_json_path):
    with open(train_json_path) as f:
            data = json.load(f)
    train = pd.DataFrame(data)
    train_en = train[train['q_lang']=="en"]
    with open(val_json_path) as f:
            data = json.load(f)
    val =  pd.DataFrame(data)
    val_en = val[val['q_lang']=="en"]
    img2idx = {}
    df = train_en.append(val_en)
    df_imgs = df['img_name'].unique().tolist()

    for i, row in tqdm(df.iterrows()):
        img_name = row['img_name']
        img_id = df_imgs.index(img_name)  # starts from 0
        if img_name not in img2idx:
            img2idx[img_name] = img_id
        else:
            assert img2idx[img_name] == img_id

    with open(out_json_path, 'w') as f:
        json.dump(img2idx, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create img2idx.json.")
    parser.add_argument("train_path", type=str, help="Path to train json file")
    parser.add_argument("val_path", type=str, help="Path to val json file")
    parser.add_argument("out_path", type=str, help="Path to output file")
    args = parser.parse_args()
    create_img2idx(args.train_path, args.val_path, args.out_path)
