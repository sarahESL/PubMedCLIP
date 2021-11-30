import numpy as np
import torch, math
from core.evaluate import accuracy

class Combiner:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.type = cfg.TRAIN.COMBINER.TYPE
        self.device = device
        self.epoch_number = cfg.TRAIN.N_EPOCH
        self.func = torch.nn.Softmax(dim=1)
        self.initilize_all_parameters()

    def initilize_all_parameters(self):
        self.alpha = 0.2
        if self.epoch_number in [90, 180]:
            self.div_epoch = 100 * (self.epoch_number // 100 + 1)
        else:
            self.div_epoch = self.epoch_number

    def reset_epoch(self, epoch):
        self.epoch = epoch
    
    def forward(self, model, criterion, image, label, meta, **kwargs):
        return eval("self.{}".format(self.type))(
            model, criterion, image, label, meta, **kwargs
        )

    def default(self, model, criterion, image, label, **kwargs):
        image, label = image.to(self.device), label.to(self.device)
        output = model(image)
        loss = criterion(output, label)
        now_result = torch.argmax(self.func(output), 1)
        now_acc = accuracy(now_result.cpu().numpy(), label.cpu().numpy())[0]

        return loss, now_acc

    def bbn_mix(self, model, criterion, image_data, question, answer, answer_target, meta, **kwargs):
        image_a, image_b = image_feature.to(self.device), meta["sample_image"].to(self.device)
        label_a, label_b = answer.to(self.device), meta["sample_answer"].to(self.device)

        feature_a, feature_b = (
            model(image_a, feature_cb=True),
            model(image_b, feature_rb=True),
        )

        l = 1 - ((self.epoch - 1) / self.div_epoch) ** 2  # parabolic decay-alpha

        mixed_feature = 2 * torch.cat((l * feature_a, (1-l) * feature_b), dim=1)
        output = model(mixed_feature, classifier_flag=True)
        loss = l * criterion(output, label_a) + (1 - l) * criterion(output, label_b)

        now_result = torch.argmax(self.func(output), 1)
        now_acc = (
                l * accuracy(now_result.cpu().numpy(), label_a.cpu().numpy())[0]
                + (1 - l) * accuracy(now_result.cpu().numpy(), label_b.cpu().numpy())[0]
        )

        return loss, now_acc

