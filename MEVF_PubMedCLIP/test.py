"""
This code is modified based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa) by Xuan B. Nguyen
"""
import argparse
import torch
from torch.utils.data import DataLoader
import dataset_RAD
import dataset_SLAKE
import base_model
import utils
import pandas as pd
import os
import json
answer_types = ['CLOSED', 'OPEN', 'ALL']
quesntion_types = ['COUNT', 'COLOR', 'ORGAN', 'PRES', 'PLANE', 'MODALITY', 'POS', 'ABN', 'SIZE', 'OTHER', 'ATTRIB']
def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores, logits

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble', type=bool, default=False,
                        help='ensemble flag. If True, generate a logit file which is used in the ensemble part')
    # MODIFIABLE MEVF HYPER-PARAMETERS--------------------------------------------------------------------------------
    # Model loading/saving
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--input', type=str, default='saved_models/SAN_MEVF',
                        help='input file directory for loading a model')
    parser.add_argument('--output', type=str, default='results',
                        help='output file directory for saving VQA answer prediction file')
    # Utilities
    parser.add_argument('--epoch', type=int, default=19,
                        help='the best epoch')

    # Gradient accumulation
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')

    # Choices of Attention models
    parser.add_argument('--model', type=str, default='SAN', choices=['BAN', 'SAN'],
                        help='the model we use')

    # Choices of RNN models
    parser.add_argument('--rnn', type=str, default='LSTM', choices=['LSTM', 'GRU'],
                        help='the RNN we use')

    # BAN - Bilinear Attention Networks
    parser.add_argument('--gamma', type=int, default=2,
                        help='glimpse in Bilinear Attention Networks')
    parser.add_argument('--use_counter', action='store_true', default=False,
                        help='use counter module')

    # SAN - Stacked Attention Networks
    parser.add_argument('--num_stacks', default=2, type=int,
                        help='num of stacks in Stack Attention Networks')

    # Utilities - gpu
    parser.add_argument('--gpu', type=int, default=0,
                        help='specify index of GPU using for training, to use CPU: -1')

    # Question embedding
    parser.add_argument('--op', type=str, default='c',
                        help='concatenated 600-D word embedding')

    # Joint representation C dimension
    parser.add_argument('--num_hid', type=int, default=1024,
                        help='dim of joint semantic features')

    # Activation function + dropout for classification module
    parser.add_argument('--activation', type=str, default='relu', choices=['relu'],
                        help='the activation to use for final classifier')
    parser.add_argument('--dropout', default=0.5, type=float, metavar='dropout',
                        help='dropout of rate of final classifier')

    # Train with RAD
    parser.add_argument('--use_RAD', action='store_true', default=False,
                        help='Using TDIUC dataset to train')
    parser.add_argument('--RAD_dir', type=str,
                        help='RAD dir')
    parser.add_argument('--use_SLAKE', action='store_true', default=False,
                        help='Using TDIUC dataset to train')
    parser.add_argument('--SLAKE_dir', type=str,
                        help='RAD dir')

    # Optimization hyper-parameters
    parser.add_argument('--eps_cnn', default=1e-5, type=float, metavar='eps_cnn',
                        help='eps - batch norm for cnn')
    parser.add_argument('--momentum_cnn', default=0.05, type=float, metavar='momentum_cnn',
                        help='momentum - batch norm for cnn')

    # input visual feature dimension
    parser.add_argument('--feat_dim', default=64, type=int,
                        help='visual feature dim')

    parser.add_argument('--feat_dim_clip', default=576, type=int,
                        help='visual feature dim when clip included')
    # Auto-encoder component hyper-parameters
    parser.add_argument('--autoencoder', action='store_true', default=False,
                        help='End to end model?')
    parser.add_argument('--ae_model_path', type=str, default='pretrained_ae.pth',
                        help='the maml_model_path we use')

    # MAML component hyper-parameters
    parser.add_argument('--maml', action='store_true', default=False,
                        help='End to end model?')
    parser.add_argument('--maml_model_path', type=str, default='pretrained_maml.weights',
                        help='the maml_model_path we use')
    parser.add_argument('--clip', action='store_true', default=False,
                        help='Use clip or not.')
    parser.add_argument('--clip_org', action='store_true', default=False,
                        help='Use original clip or not.')
    parser.add_argument('--clip_path', type=str, default="path/to/fine-tuned/PubMedCLIP",
                        help='the clip_model_path we use')
    parser.add_argument('--clip_vision_encoder', type=str, default="ViT-B/32",
                        help='Use transformer or resnet')

    # Return args
    args = parser.parse_args()
    return args
# Load questions


def get_question(q, dataloader):
    q = q.squeeze(0)
    str = []
    dictionary = dataloader.dataset.dictionary
    for i in range(q.size(0)):
        str.append(dictionary.idx2word[q[i]] if q[i] < len(dictionary.idx2word) else '_')
    return ' '.join(str)


# Load answers
def get_answer(p, dataloader):
    _m, idx = p.max(1)
    return dataloader.dataset.label2ans[idx.item()]


# Logit computation (for train, test or evaluate)
def get_result(model, dataloader, device, args):
    targeted_results = {"image_name": [], "question": [], "answer": [], "predicted_answer": []}
    keys = ['count', 'real', 'true', 'real_percent', 'score', 'score_percent']
    question_types_result = dict((i, dict((j, dict((k, 0.0) for k in keys)) for j in quesntion_types)) for i in answer_types)
    result = dict((i, dict((j, 0.0) for j in keys)) for i in answer_types)
    #targeted_imgs = ["synpic31955.jpg", "synpic17738.jpg", "synpic32970.jpg", "synpic25821.jpg", "synpic31955.jpg", "synpic24878.jpg"]  # RAD
    #targeted_imgs = ["synpic40464.jpg"]
    targeted_imgs = ["synpic40464.jpg", "synpic47974.jpg", "synpic42805.jpg"]
    # targeted_imgs = ["xmlab442/source.jpg", "xmlab386/source.jpg", "xmlab102/source.jpg", "xmlab385/source.jpg"]   #SLAKE
    with torch.no_grad():
        for v, q, a, ans_type, q_types, p_type, image_name, question_text, answer_text in iter(dataloader):
            #if p_type[0] != "freeform":
            #    continue
            if args.maml:
                v[0] = v[0].reshape(v[0].shape[0], 84, 84).unsqueeze(1)
            if args.autoencoder:
                v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1)
            if args.clip:
                if args.clip_vision_encoder == "RN50x4":
                    v[2] = v[2].reshape(v[2].shape[0], 3, 288, 288)
                else:
                    v[2] = v[2].reshape(v[2].shape[0], 3, 250, 250)
            v[0] = v[0].to(device)
            v[1] = v[1].to(device)
            v[2] = v[2].to(device)
            q = q.to(device)
            a = a.to(device)
            # inference and get logit
            if args.autoencoder:
                features, _ = model(v, q)
            else:
                features = model(v, q)
            preds = model.classifier(features)
            final_preds = preds
            batch_score_temp, logits = compute_score_with_logits(final_preds, a.data)
            batch_score = batch_score_temp.sum()
            for i, img in enumerate(image_name):
                if img in targeted_imgs:
                    targeted_results["image_name"].append(img)
                    targeted_results["question"].append(question_text[i])
                    targeted_results["answer"].append(answer_text[i])
                    targeted_results["predicted_answer"].append(logits[i].cpu())

            # Compute accuracy for each type answer
            result[ans_type[0]]['count'] += 1.0
            result[ans_type[0]]['true'] += float(batch_score)
            result[ans_type[0]]['real'] += float(a.sum())

            result['ALL']['count'] += 1.0
            result['ALL']['true'] += float(batch_score)
            result['ALL']['real'] += float(a.sum())

        for i in answer_types:
            if result[i]['count'] != 0:
                result[i]['score'] = result[i]['true']/result[i]['count']
                result[i]['score_percent'] = round(result[i]['score']*100,1)
    return result, question_types_result, targeted_results


# Test phase
if __name__ == '__main__':
    args = parse_args()
    print(args)
    torch.backends.cudnn.benchmark = True
    args.device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")

    # Check if evaluating on TDIUC dataset or VQA dataset
    if args.use_RAD:
        dictionary = dataset_RAD.Dictionary.load_from_file(os.path.join(args.RAD_dir , 'dictionary.pkl'))
        eval_dset = dataset_RAD.VQAFeatureDataset(args.split, args, dictionary)

    if args.use_SLAKE:
        dictionary = dataset_SLAKE.Dictionary.load_from_file(os.path.join(args.SLAKE_dir, 'dictionary.pkl'))
        train_dset = dataset_SLAKE.VQASLAKEFeatureDataset('train', args,dictionary,dataroot="./data_SLAKE")
        eval_dset = dataset_SLAKE.VQASLAKEFeatureDataset('test', args,dictionary,dataroot="./data_SLAKE")
    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=utils.trim_collate)

    def save_questiontype_results(outfile_path, quesntion_types_result):
        for i in quesntion_types_result:
            pd.DataFrame(quesntion_types_result[i]).transpose().to_csv(outfile_path + '/question_type_' + i + '.csv')
    # Testing process
    def process(args, model, eval_loader):
        model_path = args.input + '/model_epoch%s.pth' % args.epoch
        print('loading %s' % model_path)
        model_data = torch.load(model_path)

        # Comment because do not use multi gpu
        # model = nn.DataParallel(model)
        model = model.to(args.device)
        model.load_state_dict(model_data.get('model_state', model_data))

        model.train(False)
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        if args.use_RAD or args.use_SLAKE:
            result, quesntion_types_result, targeted_results = get_result(model, eval_loader, args.device, args)
            outfile_path = args.output + '/' + args.input.split('/')[1]
            outfile = outfile_path + '/results.json'
            if not os.path.exists(os.path.dirname(outfile)):
                os.makedirs(os.path.dirname(outfile))
            print(result)
            print(quesntion_types_result)
            json.dump(result, open(outfile, 'w'))
            save_questiontype_results(outfile_path, quesntion_types_result)
            df = pd.DataFrame(targeted_results)
            df.to_csv("predicted_results/rad_targeted_predictions_fail.csv", index=False)  # RAD
            #df.to_csv("predicted_results/slake_targeted_predictions.csv", index=False)  # SLAKE
        return
    process(args, model, eval_loader)
