#!/usr/bin/env python

import argparse
import gc
import json
import logging
import os
import pandas as pd
from tqdm import tqdm


logging.basicConfig()
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def _imgpath(img_dir, name):
    img_path = os.path.join(img_dir, name)
    if not os.path.exists(img_path):
        return "nofile"
    return img_path

def _imgsize(img_path):
    size = os.path.getsize(img_path)
    return size



def create_jsons(train_path, validation_path, test_path, jsonpath):
    # read data
    train_df = pd.read_csv(os.path.join(train_path, "radiologytraindata.csv"))
    validation_df = pd.read_csv(os.path.join(validation_path, "radiologyvaldata.csv"))
    test_df = pd.read_csv(os.path.join(test_path, "radiologytestdata.csv"))

    assert len(train_df.columns) == 3 
    assert len(validation_df.columns) == 3 
    assert len(test_df.columns) == 3 

    assert "id" and "name" and "caption" in train_df.columns
    assert "id" and "name" and "caption" in validation_df.columns
    assert "id" and "name" and "caption" in test_df.columns

    # convert df rows to dict
    logger.info("Converting each row in dataframe to dictionary...")
    train_df.drop(columns=['id'], inplace=True)
    validation_df.drop(columns=['id'], inplace=True)
    test_df.drop(columns=['id'], inplace=True)

    ## add full image paths
    train_image_dir = os.path.join(os.path.join(train_path, "radiology", "images"))
    validation_image_dir = os.path.join(os.path.join(validation_path, "radiology", "images"))
    test_image_dir = os.path.join(os.path.join(test_path, "radiology", "images"))

    train_df['name'] = train_df['name'].apply(lambda x: _imgpath(train_image_dir, x))
    validation_df['name'] = validation_df['name'].apply(lambda x: _imgpath(validation_image_dir, x))
    test_df['name'] = test_df['name'].apply(lambda x: _imgpath(test_image_dir, x))

    ### drop files that don't exist: for some names in csv files, the actual image does not exist
    train_df = train_df[train_df['name'] != "nofile"]
    validation_df = validation_df[validation_df['name'] != "nofile"]
    test_df = test_df[test_df['name'] != "nofile"]

    ### drop zero bytes images
    train_df['imagesize'] = train_df['name'].apply(lambda x: _imgsize(x))
    validation_df['imagesize'] = validation_df['name'].apply(lambda x: _imgsize(x))
    test_df['imagesize'] = test_df['name'].apply(lambda x: _imgsize(x))

    train_df = train_df[train_df['imagesize'] != 0]
    validation_df = validation_df[validation_df['imagesize'] != 0]
    test_df = test_df[test_df['imagesize'] != 0]

    train_df.drop(columns=['imagesize'], inplace=True)
    validation_df.drop(columns=['imagesize'], inplace=True)
    test_df.drop(columns=['imagesize'], inplace=True)

    train_df.rename(columns={"name": "image_path"}, inplace=True)
    validation_df.rename(columns={"name": "image_path"}, inplace=True)
    test_df.rename(columns={"name": "image_path"}, inplace=True)

    ## convert to dict
    train_dict = train_df.to_dict('index')
    validation_dict = validation_df.to_dict('index')
    test_dict = test_df.to_dict('index')

    del [[train_df, validation_df, test_df]]
    gc.collect()

    # Dump to json
    ## train
    logger.info("Dumping json data for train dataset...")
    with open(os.path.join(jsonpath, "train_dataset.json"), 'w') as f:
        for row in tqdm(train_dict):
            json.dump(train_dict[row], f)
            f.write("\n")
    ## validation
    logger.info("Dumping json data for validation dataset...")
    with open(os.path.join(jsonpath, "validation_dataset.json"), 'w') as f:
        for row in tqdm(validation_dict):
            json.dump(validation_dict[row], f)
            f.write("\n")
    ## test
    logger.info("Dumping json data for test dataset...")
    with open(os.path.join(jsonpath, "test_dataset.json"), 'w') as f:
        for row in tqdm(test_dict):
            json.dump(test_dict[row], f)
            f.write("\n")

    logger.info("Jsons are successfly created!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create input jsons using train, validation and test sets for medCLIP.")
    parser.add_argument(type=str, dest="train", help="Path to train directory containing images folder and csv file.")
    parser.add_argument(type=str, dest="validation", help="Path to validation directory containing images folder and csv file.")
    parser.add_argument(type=str, dest="test", help="Path to test directory containing images folder and csv file.")
    parser.add_argument(type=str, dest="jsonpath", help="Path to json directory to dump output jsons.")

    train_path = parser.parse_args().train
    validation_path = parser.parse_args().validation
    test_path = parser.parse_args().test
    json_path = parser.parse_args().jsonpath

    create_jsons(train_path, validation_path, test_path, json_path)
