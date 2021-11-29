import _init_paths
from core.evaluate import AverageMeter
import numpy as np
import torch
import time
import clip


def valid_model(dataLoader, model, criterion_img, criterion_txt, cfg, device):
    model.eval()
    all_loss = AverageMeter()
    
    with torch.no_grad():
        for i, (image, caption) in enumerate(dataLoader):
            images = torch.stack([img for img in image], dim=0).to(device)
            captions = clip.tokenize(caption).to(device)

            logits_per_image, logits_per_text = model(images, captions)
            logits_per_image *= (np.exp(0.01) / np.exp(0.07))
            logits_per_text *= (np.exp(0.01) / np.exp(0.07))

            ground_truth = torch.arange(cfg.TEST.BATCH_SIZE, dtype=torch.long, device=device)
            total_loss = (criterion_img(logits_per_image, ground_truth) + criterion_txt(logits_per_text, ground_truth)) / 2
            all_loss.update(total_loss.data.item(), len(caption))

        return all_loss.avg
