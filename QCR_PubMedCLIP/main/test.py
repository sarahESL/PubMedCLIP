# -*- coding: utf-8 -*-#

#-------------------------------------------------------------------------------
# Name:         train
# Description:
# Author:       Boliu.Kelvin, Sedigheh Eslami
#-------------------------------------------------------------------------------
import os
import time
import torch
from utils import utils
from datetime import datetime
from pathlib import Path
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import pandas as pd


def compute_score_with_logits(logits, labels):
    # batch_open_score_temp, open_logits = compute_score_with_logits(preds_open, a_open.data)
    func = torch.nn.Softmax(dim=1)
    logits = func(logits)
    logits = torch.max(logits, 1)[1].data  # argmax
    print('compute_score_with_logits, logits shape: ', logits.shape)
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores, logits

def get_time_stamp():
    ct = time.time()
    local_time = time.localtime(ct)
    data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s.%03d" % (data_head, data_secs)
    return time_stamp

# Train phase
def test(cfg, model, question_model, eval_loader, n_unique_close, device, s_opt=None, s_epoch=0):
    model = model.to(device)  # BAN_model
    question_model = question_model.to(device)
    utils.create_dir(cfg.TEST.RESULT_DIR)

    # Evaluation
    eval_score, open_score, close_score = evaluate_classifier(model, question_model, eval_loader,
                                                              cfg, n_unique_close, device,
                                                              cfg.TEST.RESULT_DIR)


def load_pickle(fp: str):
    import pickle
    with open(fp, 'rb') as f:
        return pickle.load(f)


# Evaluation
def evaluate_classifier(model, pretrained_model, dataloader, cfg, n_unique_close, device,
                        result_dir):
    score = 0
    total = 0
    open_ended = 0.  #'OPEN'
    score_open = 0.

    closed_ended = 0.  #'CLOSED'
    score_close = 0.
    model.eval()

    correct_results = {
        "image_name": [],
        "question": [],
        "answer": [],
        "predicted_answer": [],
        "answer_type": [],
        "predicted_answer_type": []
    }
    incorrect_results = {
        "image_name": [],
        "question": [],
        "answer": [],
        "predicted_answer": [],
        "answer_type": [],
        "predicted_answer_type": []
    }
    q_img = []
    with torch.no_grad():
        for i, (v, q, a, answer_type, question_type, phrase_type, answer_target, image_name,
                question_text, answer_text) in enumerate(dataloader):

            if cfg.TRAIN.VISION.MAML:
                v[0] = v[0].reshape(v[0].shape[0], 84, 84).unsqueeze(1)
                v[0] = v[0].to(device)
            if cfg.TRAIN.VISION.AUTOENCODER:
                v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1)
                v[1] = v[1].to(device)
            if cfg.TRAIN.VISION.CLIP:
                if cfg.TRAIN.VISION.CLIP_VISION_ENCODER == "RN50x4":
                    v[2] = v[2].reshape(v[2].shape[0], 3, 288, 288)
                else:
                    v[2] = v[2].reshape(v[2].shape[0], 3, 250, 250)
                v[2] = v[2].to(device)
            if cfg.TRAIN.VISION.OTHER_MODEL:
                v = v.to(device)

            q[0] = q[0].to(device)
            if cfg.TRAIN.QUESTION.CLIP:
                q[1] = q[1].to(device)
            a = a.to(device)

            if cfg.TRAIN.VISION.AUTOENCODER:
                last_output_close, last_output_open, a_close, a_open, decoder, indexs_open, indexs_close = model.forward_classify(
                    v, q, a, pretrained_model, n_unique_close)
            else:
                last_output_close, last_output_open, a_close, a_open, indexs_open, indexs_close = model.forward_classify(
                    v, q, a, pretrained_model, n_unique_close)
            # self.close_classifier = SimpleClassifier(cfg.TRAIN.QUESTION.CLS_HID_DIM, cfg.TRAIN.QUESTION.CLS_HID_DIM * 2, dataset.num_close_candidates, cfg)
            # self.open_classifier = SimpleClassifier(cfg.TRAIN.QUESTION.CLS_HID_DIM, cfg.TRAIN.QUESTION.CLS_HID_DIM * 2, dataset.num_open_candidates, cfg)

            preds_close, preds_open = model.classify(last_output_close, last_output_open)
            print("evaluate_classifier, preds_close shape: ",
                  preds_close.shape)  # torch.Size([5, 56])
            print("evaluate_classifier, preds_open shape: ",
                  preds_open.shape)  # torch.Size([3, 431])

            batch_close_score = 0.
            batch_open_score = 0.
            if preds_close.shape[0] != 0:
                # preds_close: prediction
                # a_close.data: gold answers, 1-hot
                batch_close_score_temp, close_logits = compute_score_with_logits(
                    preds_close, a_close.data)
                close_correct = (batch_close_score_temp == 1).nonzero(as_tuple=True)[0].tolist()
                batch_close_score = batch_close_score_temp.sum()
            if preds_open.shape[0] != 0:
                batch_open_score_temp, open_logits = compute_score_with_logits(
                    preds_open, a_open.data)
                open_correct = (batch_open_score_temp == 1).nonzero(as_tuple=True)[0].tolist()
                batch_open_score = batch_open_score_temp.sum()

            score += batch_close_score + batch_open_score

            size = q[0].shape[0]
            total += size  # batch number

            open_ended += preds_open.shape[0]
            score_open += batch_open_score

            closed_ended += preds_close.shape[0]
            score_close += batch_close_score
            # indexes_{type} is the indexes inside the batch that are {type}-ended
            # indexes_close: [0, 1, 3, 4, 5, 7]
            # indexes_open: [2, 6]
            assert len(indexs_close) + len(indexs_open) == len(image_name)
            assert len(close_correct) + len(open_correct) <= len(image_name)  # batch size

            # PubMedCLIP/QCR_PubMedCLIP/data/data_rad/cache/close_label2ans.pkl
            # PubMedCLIP/QCR_PubMedCLIP/data/data_rad/cache/open_label2ans.pkl

            cache_root = './data/data_rad/cache'
            close_l2a = load_pickle(Path(cache_root) / 'close_label2ans.pkl')
            open_l2a = load_pickle(Path(cache_root) / 'open_label2ans.pkl')

            close_incorrect = [i for i in range(len(indexs_close)) if i not in close_correct]
            open_incorrect = [i for i in range(len(indexs_open)) if i not in open_correct]
            # print("close")
            # print("\tindices", indexs_close)
            # print("\tcorrect: ", close_correct)
            # print("\tincorrect: ", close_incorrect)
            # print("open")
            # print("\tindices", indexs_open)
            # print("\tcorrect: ", open_correct)
            # print("\tincorrect: ", open_incorrect)

            # Bugfix
            # close
            #         (batch) indices [0, 1, 2, 6]
            #         correct:  [2, 3]      -> [2, 6]
            #         incorrect:  [0, 1]    -> [0, 1]
            # open
            #         (batch) indices [3, 4, 5, 7]
            #         correct:  [0, 2, 3] -> [3, 5, 7]
            #         incorrect:  [1]     -> [4]

            for i in close_correct:
                ind = indexs_close[i]

                pred_label = close_logits[i].cpu().item()
                pred_answer = close_l2a[pred_label]
                correct_results["predicted_answer"].append(pred_answer)
                correct_results["image_name"].append(image_name[ind])
                correct_results["question"].append(question_text[ind])
                correct_results["answer"].append(answer_text[ind])
                correct_results["predicted_answer_type"].append("CLOSED")
                correct_results["answer_type"].append(answer_type[ind])
            for i in close_incorrect:
                ind = indexs_close[i]
                pred_label = close_logits[i].cpu().item()
                pred_answer = close_l2a[pred_label]
                incorrect_results["image_name"].append(image_name[ind])
                incorrect_results["question"].append(question_text[ind])
                incorrect_results["answer"].append(answer_text[ind])
                incorrect_results["predicted_answer"].append(pred_answer)
                incorrect_results["predicted_answer_type"].append("CLOSED")
                incorrect_results["answer_type"].append(answer_type[ind])
            for i in open_correct:
                ind = indexs_open[i]

                pred_label = open_logits[i].cpu().item()
                pred_answer = open_l2a[pred_label]

                correct_results["image_name"].append(image_name[ind])
                correct_results["question"].append(question_text[ind])
                correct_results["answer"].append(answer_text[ind])
                correct_results["predicted_answer"].append(pred_answer)
                correct_results["predicted_answer_type"].append("OPEN")
                correct_results["answer_type"].append(answer_type[ind])

            for i in open_incorrect:
                ind = indexs_open[i]

                pred_label = open_logits[i].cpu().item()
                pred_answer = open_l2a[pred_label]
                incorrect_results["image_name"].append(image_name[ind])
                incorrect_results["question"].append(question_text[ind])
                incorrect_results["answer"].append(answer_text[ind])
                incorrect_results["predicted_answer"].append(pred_answer)
                incorrect_results["predicted_answer_type"].append("OPEN")
                incorrect_results["answer_type"].append(answer_type[ind])



    try:
        score = 100 * score / total
    except ZeroDivisionError:
        score = 0
    try:
        open_score = 100 * score_open / open_ended
    except ZeroDivisionError:
        open_score = 0
    try:
        close_score = 100 * score_close / closed_ended
    except ZeroDivisionError:
        close_score = 0
    print(total, open_ended, closed_ended)
    print('[Validate] Val_Acc:{:.6f}%  |  Open_ACC:{:.6f}%   |  Close_ACC:{:.6f}%'.format(
        score, open_score, close_score))
    df = pd.DataFrame(correct_results)
    df.to_csv(f"{result_dir}/correct_predictions.csv", index=False)
    inc_df = pd.DataFrame(incorrect_results)
    inc_df.to_csv(f"{result_dir}/incorrect_predictions.csv", index=False)

    return score, open_score, close_score
