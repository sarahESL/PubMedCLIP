#!/usr/bin/env python


import argparse
from PIL import Image
import pandas as pd
import os
import json
import numpy as np
import pickle
from tqdm import tqdm


def imageresize(img2idx_jsonpath, img_folderpath, reshape_size, out_path, channels):
    with open(img2idx_jsonpath) as f:
        img2idx = json.load(f)
    
    if channels == 3:
        imgs = np.ndarray(shape=(len(img2idx), reshape_size, reshape_size, 3), dtype=float)
    else:
        imgs = np.ndarray(shape=(len(img2idx), reshape_size, reshape_size, 1), dtype=float)

    for imgid, idx in tqdm(img2idx.items()):
        if ".jpg" in imgid:
            imgpath = os.path.join(img_folderpath, imgid)
        else:
            imgpath = os.path.join(img_folderpath, f"{imgid}.jpg")
        if os.path.exists(imgpath):
            if channels == 3:
                img = Image.open(imgpath).convert('RGB')
            else:
                img = Image.open(imgpath).convert('L')
        else:
            raise ValueError(f"Image path is not correct: {imgpath}")
        resized = img.resize((reshape_size, reshape_size))
        normalized = np.array(resized) / 255
        if channels == 3:
            normalized = normalized.reshape((reshape_size, reshape_size, 3))
        else:
            normalized = normalized.reshape((reshape_size, reshape_size, 1))
        imgs[idx] = normalized


    with open(out_path, 'wb') as f:
        pickle.dump(imgs, f)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create image84x84 and image128x128 for all train, val, test images.")
    parser.add_argument("img2idx_json", type=str, help="Path to img2idx.json file")
    parser.add_argument("allimgs_folder", type=str, help="Path to all images folder")
    parser.add_argument("size", type=int, help="Reshape size")
    parser.add_argument("out_path", type=str, help="Path to output file for reshaped images")
    parser.add_argument("channels", type=int, help="Path to output file for reshaped images")
    args = parser.parse_args()
    imageresize(args.img2idx_json, args.allimgs_folder, args.size, args.out_path, args.channels)
