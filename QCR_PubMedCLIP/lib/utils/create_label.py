

# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         process_dataset
# Description:  convert original .txt file to train.json and validate.json
# Author:       Boliu.Kelvin, Sedigheh Eslami
# Date:         2020/4/5
#-------------------------------------------------------------------------------

import argparse
import pandas as pd
import os
import sys
import json
import numpy as np
import re
import _pickle as cPickle

contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": \
    "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've", \
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": \
    "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've": \
    "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": \
    "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": \
    "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", \
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": \
    "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": \
    "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've": \
    "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", \
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": \
    "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": \
    "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt": \
    "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": \
    "shouldn't've", "somebody'd": "somebodyd", "somebodyd've": \
    "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": \
    "somebody'll", "somebodys": "somebody's", "someoned": "someone'd", \
    "someoned've": "someone'd've", "someone'dve": "someone'd've", \
    "someonell": "someone'll", "someones": "someone's", "somethingd": \
    "something'd", "somethingd've": "something'd've", "something'dve": \
    "something'd've", "somethingll": "something'll", "thats": \
    "that's", "thered": "there'd", "thered've": "there'd've", \
    "there'dve": "there'd've", "therere": "there're", "theres": \
    "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve": \
    "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": \
    "they've", "twas": "'twas", "wasnt": "wasn't", "wed've": \
    "we'd've", "we'dve": "we'd've", "weve": "we've", "werent":
    "weren't", "whatll": "what'll", "whatre": "what're", "whats": \
    "what's", "whatve": "what've", "whens": "when's", "whered": \
    "where'd", "wheres": "where's", "whereve": "where've", "whod": \
    "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": \
    "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": \
    "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": \
    "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": \
    "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll": \
    "you'll", "youre": "you're", "youve": "you've"}
manual_map = { 'none': '0',
              'zero': '0',
              'one': '1',
              'two': '2',
              'three': '3',
              'four': '4',
              'five': '5',
              'six': '6',
              'seven': '7',
              'eight': '8',
               'nine': '9',
              'ten': '10'}
articles = ['a', 'an', 'the']
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [';', r"/", '[', ']', '"', '{', '}',
                '(', ')', '=', '+', '\\', '_', '-',
                '>', '<', '@', '`', ',', '?', '!']

def process_punctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) \
           or (re.search(comma_strip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText

def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText

def preprocess_answer(answer):
    answer = str(answer)
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(',', '').replace('x ray', 'xray')
    return answer

def filter_answers(train_qa_pairs, val_qa_pairs, min_occurence):
    """This will change the answer to preprocessed version
    """
    occurence = {}
    qa_pairs = train_qa_pairs.append(val_qa_pairs)
    qa_pairs['answer'] = qa_pairs['answer'].apply(lambda x: str(x))

    for id, row in qa_pairs.iterrows(): # row:[id,ques,ans]
        gtruth = row['answer']
        gtruth = ' '.join(gtruth.split())
        if gtruth not in occurence:
            occurence[gtruth] = set()
        occurence[gtruth].add(row['qid'])
    for answer in list(occurence):
        if len(occurence[answer]) < min_occurence:
            occurence.pop(answer)

    print('Num of answers that appear >= %d times: %d' % (
        min_occurence, len(occurence)))
    return occurence

def filter_answers_open_close(train_qa_pairs, val_qa_pairs, min_occurence):
    """This will change the answer to preprocessed version
    """
    occurence_open = {}
    occurence_close = {}
    qa_pairs = train_qa_pairs.append(val_qa_pairs)
    qa_pairs['answer'] = qa_pairs['answer'].apply(lambda x: str(x))
    qa_pairs_open = qa_pairs[qa_pairs['answer_type']=="OPEN"]
    qa_pairs_close = qa_pairs[qa_pairs['answer_type']=="CLOSED"]

    for id, row in qa_pairs_open.iterrows(): # row:[id,ques,ans]
        gtruth = row['answer']
        gtruth = ' '.join(gtruth.split())
        if gtruth not in occurence_open:
            occurence_open[gtruth] = set()
        occurence_open[gtruth].add(row['qid'])
    for answer in list(occurence_open):
        if len(occurence_open[answer]) < min_occurence:
            occurence_open.pop(answer)

    print('Num of open answers that appear >= %d times: %d' % (
        min_occurence, len(occurence_open)))
    for id, row in qa_pairs_close.iterrows(): # row:[id,ques,ans]
        gtruth = row['answer']
        gtruth = ' '.join(gtruth.split())
        if gtruth not in occurence_close:
            occurence_close[gtruth] = set()
        occurence_close[gtruth].add(row['qid'])
    for answer in list(occurence_close):
        if len(occurence_close[answer]) < min_occurence:
            occurence_close.pop(answer)

    print('Num of close answers that appear >= %d times: %d' % (
        min_occurence, len(occurence_close)))
    return occurence_open, occurence_close

def create_ans2label(occurence, train_qa_pairs, val_qa_pairs, filename="trainval", root='data'):
    """Note that this will also create label2ans.pkl at the same time

    occurence: dict {answer -> whatever}
    name: prefix of the output file
    cache_root: str
    """
    df = train_qa_pairs.append(val_qa_pairs)
    df['answer'] = df['answer'].apply(lambda x: str(x).lower())
    close_answers = df[df['answer_type']=="CLOSED"]['answer'].unique()
    open_answers = df[df['answer_type']=="OPEN"]['answer'].unique()
    intersection = set(close_answers).intersection(set(open_answers))
    ans2label = {}
    label2ans = []
    close_ans2label = {}
    close_label2ans = []
    open_ans2label = {}
    open_label2ans = []
    label = 0
    for answer in close_answers:
        label2ans.append(answer)
        ans2label[answer] = label
        close_label2ans.append(answer)
        close_ans2label[answer] = label
        label += 1

    open_label = 0
    for answer in open_answers:
        if answer in ans2label:
            ans = answer + "#"
            label2ans.append(ans)
            ans2label[ans] = label
            open_label2ans.append(ans)
            open_ans2label[ans] = open_label
        else:
            label2ans.append(answer)
            ans2label[answer] = label
            open_label2ans.append(answer)
            open_ans2label[answer] = open_label
        label += 1
        open_label += 1

    print('ans2lab', len(ans2label))
    print('lab2abs', len(label2ans))

    if not os.path.exists(os.path.join(root, "cache")):
        os.mkdir(os.path.join(root, "cache"))

    file = os.path.join(root, "cache", 'trainval_ans2label.pkl')
    cPickle.dump(ans2label, open(file, 'wb'))
    file = os.path.join(root, "cache", f'trainval_label2ans.pkl')
    cPickle.dump(label2ans, open(file, 'wb'))

    file = os.path.join(root, "cache", 'close_ans2label.pkl')
    cPickle.dump(close_ans2label, open(file, 'wb'))
    file = os.path.join(root, "cache", f'close_label2ans.pkl')
    cPickle.dump(close_label2ans, open(file, 'wb'))

    file = os.path.join(root, "cache", 'open_ans2label.pkl')
    cPickle.dump(open_ans2label, open(file, 'wb'))
    file = os.path.join(root, "cache", f'open_label2ans.pkl')
    cPickle.dump(open_label2ans, open(file, 'wb'))
    return ans2label, intersection

def compute_target(answers_dset, ans2label, intersection, name, image_id_col="id", root='data', with_type=False):
    """Augment answers_dset with soft score as label

    ***answers_dset should be preprocessed***

    Write result into a cache file
    """
    target = []
    count = 0
    answers_dset['answer'] = answers_dset['answer'].apply(lambda x: str(x).lower())
    close_answers = answers_dset[answers_dset['answer_type']=="CLOSED"]['answer'].unique()
    open_answers = answers_dset[answers_dset['answer_type']=="OPEN"]['answer'].unique()
    for id,qa_pair in answers_dset.iterrows():
        answers = ' '.join(qa_pair['answer'].split())
        answer_type = qa_pair['answer_type']
        if (answers in intersection) and (answer_type == "OPEN"):
            answers = answers + "#"

        labels = []
        scores = []
        if answers in ans2label:
            scores.append(1.)
            labels.append(ans2label[answers])
        if with_type:
            if qa_pair['answer_type'].strip().lower() == "closed":
                a_type = 0
            elif qa_pair['answer_type'].strip().lower() == "open":
                a_type = 1
            else:
                raise ValueError(f"Unsupported answer type: {qa_pair['answer_type']}!")
            target.append({
                'qid': qa_pair['qid'],
                'img_name': qa_pair[image_id_col],
                'labels': labels,
                'scores': scores,
                'type': a_type
                })
        else:
            target.append({
                'qid': qa_pair['qid'],
                'img_name': qa_pair[image_id_col],
                'labels': labels,
                'scores': scores
                })
    if with_type:
        file = os.path.join(root, "cache", name+'_openclose_target.pkl')
    else:
        file = os.path.join(root, "cache", name+'_target.pkl')
    cPickle.dump(target, open(file, 'wb'))
    return target


if __name__ == '__main__' :
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

    train_path = os.path.join(data, train_file)
    with open(train_path) as f:
        d = json.load(f)
    train = pd.DataFrame(d)

    validate_path = os.path.join(data, test_file)
    with open(validate_path) as f:
        dd = json.load(f)
    val = pd.DataFrame(dd)
    img_col = "image_name"
    if dataset.lower() in ["slake", "vqa_slake", "vqa-slake"]:
        train = train[train['q_lang']=="en"]
        val = val[val['q_lang']=="en"]
        img_col = "img_name"

    train_qa_pairs = train[[img_col, 'qid', 'answer', 'answer_type']]
    val_qa_pairs = val[[img_col, 'qid', 'answer', 'answer_type']]

    occurence = filter_answers(train_qa_pairs, val_qa_pairs, 0)  # select the answer with frequence over min_occurence
    open_occurence, close_occurence = filter_answers_open_close(train_qa_pairs, val_qa_pairs, 0)  # select the answer with frequence over min_occurence

    label_path = data + 'cache/trainval_ans2label.pkl'
    if os.path.isfile(label_path):
        print('found %s' % label_path)
        total_ans2label = cPickle.load(open(label_path, 'rb'))
    else:
        total_ans2label, intersection = create_ans2label(occurence, train_qa_pairs, val_qa_pairs, filename="trainval", root=data)     # create ans2label and label2ans

    compute_target(train_qa_pairs, total_ans2label, intersection, 'train', img_col, data) #dump train target to .pkl {question,image_name,labels,scores}
    compute_target(train_qa_pairs, total_ans2label, intersection, 'train', img_col, data, with_type=True) #dump train target to .pkl {question,image_name,labels,scores}

    compute_target(val_qa_pairs, total_ans2label, intersection, 'test', img_col, data)   #dump validate target to .pkl {question,image_name,labels,scores}
    compute_target(val_qa_pairs, total_ans2label, intersection, 'test', img_col, data, with_type=True)   #dump validate target to .pkl {question,image_name,labels,scores}

    print("Process finished successfully!")
