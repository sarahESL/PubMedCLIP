"""
This code is modified based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa) by Xuan B. Nguyen
"""
import os
import time
import torch
import utils
import torch.nn as nn
from trainer import Trainer
import json
warmup_updates = 4000

answer_types = ['CLOSED', 'OPEN', 'ALL']
quesntion_types = ['COUNT', 'COLOR', 'ORGAN', 'PRES', 'PLANE', 'MODALITY', 'POS', 'ABN', 'SIZE', 'OTHER', 'ATTRIB']
# Kaiming normalization initialization
def init_weights(m):
    if type(m) == nn.Linear:
        with torch.no_grad():
            torch.nn.init.kaiming_normal_(m.weight)

# VQA score computation
def compute_score_with_logits(logits, labels):
    if labels.shape[0] == 0:     # sometimes, all samples in the batch are either open or close
                                 # hence, the labels and logits is empty
        scores = torch.zeros(*labels.size()).to(logits.device)
        return scores
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


# Train phase
def train(args, model, train_loader, eval_loader, num_epochs, output, opt=None, s_epoch=0):
    device = args.device
    # Scheduler learning rate
    lr_default = args.lr
    lr_decay_step = 2
    lr_decay_rate = .75
    best_model = ""
    lr_decay_epochs = range(10,20,lr_decay_step) if eval_loader is not None else range(10,20,lr_decay_step)
    gradual_warmup_steps = [0.5 * lr_default, 1.0 * lr_default, 1.5 * lr_default, 2.0 * lr_default]
    saving_epoch = 15    # Start point for model saving
    grad_clip = args.clip_norm

    utils.create_dir(output)

    # Adamax optimizer
    optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_default) \
        if opt is None else opt

    # Loss function
    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    ae_criterion = torch.nn.MSELoss()

    # write hyper-parameter to log file
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    logger.write(args.__repr__())
    utils.print_model(model, logger)
    logger.write('optim: adamax lr=%.4f, decay_step=%d, decay_rate=%.2f, grad_clip=%.2f' % \
        (lr_default, lr_decay_step, lr_decay_rate, grad_clip))

    # create trainer
    trainer = Trainer(args, model, criterion, optim, ae_criterion)
    update_freq = int(args.update_freq)
    wall_time_start = time.time()

    best_eval_score = 0
    # Epoch passing in training phase
    for epoch in range(s_epoch, num_epochs):
        total_loss = 0
        train_score = 0
        total_norm = 0
        count_norm = 0
        num_updates = 0
        t = time.time()
        N = len(train_loader.dataset)
        num_batches = int(N/args.batch_size + 1)
        if epoch < len(gradual_warmup_steps):
            trainer.optimizer.param_groups[0]['lr'] = gradual_warmup_steps[epoch]
            logger.write('gradual warm up lr: %.4f' % trainer.optimizer.param_groups[0]['lr'])
        elif epoch in lr_decay_epochs:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay_rate
            logger.write('decreased lr: %.4f' % trainer.optimizer.param_groups[0]['lr'])
        else:
            logger.write('lr: %.4f' % trainer.optimizer.param_groups[0]['lr'])

        # Predicting and computing score
        for i, (v, q, a, _, _, _) in enumerate(train_loader):
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
            sample = [v, q, a]

            if i < num_batches - 1 and (i + 1) % update_freq > 0:
                trainer.train_step(sample, update_params=False)
            else:
                loss, grad_norm, batch_score = trainer.train_step(sample, update_params=True)
                total_norm += grad_norm
                count_norm += 1

                total_loss += loss.item()
                train_score += batch_score
                num_updates += 1
                if num_updates % int(args.print_interval / update_freq) == 0:
                    print("Iter: {}, Loss {:.4f}, Norm: {:.4f}, Total norm: {:.4f}, Num updates: {}, Wall time: {:.2f}, ETA: {}".format(i + 1, total_loss / ((num_updates + 1)), grad_norm, total_norm, num_updates, time.time() - wall_time_start, utils.time_since(t, i / num_batches)))

        total_loss /= num_updates
        train_score = 100 * train_score / (num_updates * args.batch_size)

        # Evaluation
        if eval_loader is not None:
            print("Evaluating...")
            trainer.model.train(False)
            eval_score, bound = evaluate(model, eval_loader, args)
            trainer.model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, norm: %.4f, score: %.2f' % (total_loss, total_norm/count_norm, train_score))
        if eval_loader is not None:
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        # Save per epoch
        if epoch >= saving_epoch:
            model_path = os.path.join(output, 'model_epoch%d.pth' % epoch)
            utils.save_model(model_path, model, epoch, trainer.optimizer)
            # Save best epoch
            if eval_loader is not None and eval_score > best_eval_score:
                model_path = os.path.join(output, 'model_epoch_best.pth')
                utils.save_model(model_path, model, epoch, trainer.optimizer)
                best_eval_score = eval_score
                best_model = model


# Evaluation
def evaluate(model, dataloader, args):
    device = args.device
    score = 0
    upper_bound = 0
    num_data = 0
    with torch.no_grad():
        for v, q, a, _, _, _, _, _, _ in iter(dataloader):
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
            if args.autoencoder:
                features, _ = model(v, q)
            else:
                features = model(v, q)
            preds = model.classifier(features)
            final_preds = preds
            batch_score = compute_score_with_logits(final_preds, a.data).sum()
            score += batch_score
            upper_bound += (a.max(1)[0]).sum()
            num_data += final_preds.size(0)

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    return score, upper_bound
