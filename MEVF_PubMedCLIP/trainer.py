"""
This code is modified based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa) by Xuan B. Nguyen
"""
import torch
import utils
import contextlib
from collections import defaultdict, OrderedDict
from meters import AverageMeter, TimeMeter
class Trainer(object):
    """
    Main class for training.
    """
    def __init__(self, args, model, criterion, optimizer=None, ae_criterion = None):
        self.args = args

        # copy model and criterion on current device
        self.model = model.to(self.args.device)
        self.criterion = criterion.to(self.args.device)
        self.ae_criterion = ae_criterion.to(self.args.device)
        # initialize meters
        self.meters = OrderedDict()
        self.meters['train_loss'] = AverageMeter()
        self.meters['train_nll_loss'] = AverageMeter()
        self.meters['valid_loss'] = AverageMeter()
        self.meters['valid_nll_loss'] = AverageMeter()
        self.meters['wps'] = TimeMeter()       # words per second
        self.meters['ups'] = TimeMeter()       # updates per second
        self.meters['wpb'] = AverageMeter()    # words per batch
        self.meters['bsz'] = AverageMeter()    # sentences per batch
        self.meters['gnorm'] = AverageMeter()  # gradient norm
        self.meters['clip'] = AverageMeter()   # % of updates clipped
        self.meters['oom'] = AverageMeter()    # out of memory
        self.meters['wall'] = TimeMeter()      # wall time in seconds

        self._buffered_stats = defaultdict(lambda: [])
        self._flat_grads = None
        self._num_updates = 0
        self._optim_history = None
        self._optimizer = None
        if optimizer is not None:
            self._optimizer = optimizer

        self.total_loss = 0.0
        self.train_score = 0.0
        self.total_norm = 0.0
        self.count_norm = 0.0

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._build_optimizer()
        return self._optimizer

    def _build_optimizer(self):
        # self._optimizer = optim.build_optimizer(self.args, self.model.parameters())
        # self._optimizer =
        # self.lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self._optimizer)
        pass

    def train_step(self, sample, update_params=True):
        """Do forward, backward and parameter update."""
        # Set seed based on args.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        # seed = self.args.seed + self.get_num_updates()
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)

        # forward and backward pass
        sample = self._prepare_sample(sample)
        loss, sample_size, oom_fwd, batch_score = self._forward(sample)
        oom_bwd = self._backward(loss)

        # buffer stats and logging outputs
        # self._buffered_stats['sample_sizes'].append(sample_size)
        self._buffered_stats['sample_sizes'].append(1)
        self._buffered_stats['ooms_fwd'].append(oom_fwd)
        self._buffered_stats['ooms_bwd'].append(oom_bwd)

        # update parameters
        if update_params:
            # gather logging outputs from all replicas
            sample_sizes = self._buffered_stats['sample_sizes']
            ooms_fwd = self._buffered_stats['ooms_fwd']
            ooms_bwd = self._buffered_stats['ooms_bwd']
            ooms_fwd = sum(ooms_fwd)
            ooms_bwd = sum(ooms_bwd)

            # aggregate stats and logging outputs
            grad_denom = sum(sample_sizes)

            grad_norm = 0
            try:
                # all-reduce and rescale gradients, then take an optimization step
                grad_norm = self._all_reduce_and_rescale(grad_denom)
                self._opt()

                # update meters
                if grad_norm is not None:
                    self.meters['gnorm'].update(grad_norm)
                    self.meters['clip'].update(1. if grad_norm > self.args.clip_norm else 0.)

                self.meters['oom'].update(ooms_fwd + ooms_bwd)

            except OverflowError as e:
                self.zero_grad()
                print('| WARNING: overflow detected, ' + str(e))

            self.clear_buffered_stats()

            return loss, grad_norm, batch_score
        else:
            return None  # buffering updates

    def _forward(self, sample, eval=False):
        # prepare model and optimizer
        if eval:
            self.model.eval()
        else:
            self.model.train()
        loss = None
        oom = 0
        batch_score = 0
        if sample is not None:
            try:
                with torch.no_grad() if eval else contextlib.ExitStack():
                    answers = sample[2]
                    img_data = sample[0][1]
                    # MEVF loss computation
                    if self.args.autoencoder:
                        features, decoder = self.model(sample[0], sample[1])
                    else:
                        features = self.model(sample[0], sample[1])
                    preds = self.model.classifier(features)
                    loss = self.criterion(preds.float(), answers)
                    if self.args.autoencoder:
                        loss_ae = self.ae_criterion(img_data, decoder)
                        loss = loss + (loss_ae*self.args.ae_alpha)
                    loss /= answers.size()[0]
                    final_preds = preds
                    batch_score = compute_score_with_logits(final_preds, sample[2].data).sum()
            except RuntimeError as e:
                if not eval and 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    oom = 1
                    loss = None
                else:
                    raise e
        return loss, len(sample[0]), oom, batch_score  # TODO: Not sure about sample size, need to recheck

    def _backward(self, loss):
        oom = 0
        if loss is not None:
            try:
                # backward pass
                loss.backward()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, skipping batch')
                    oom = 1
                    self.zero_grad()
                else:
                    raise e
        return oom

    def _all_reduce_and_rescale(self, grad_denom):
        # flatten grads into a single buffer and all-reduce
        flat_grads = self._flat_grads = self._get_flat_grads(self._flat_grads)

        # rescale and clip gradients
        flat_grads.div_(grad_denom)
        grad_norm = utils.clip_grad_norm_(flat_grads, self.args.clip_norm)

        # copy grads back into model parameters
        self._set_flat_grads(flat_grads)

        return grad_norm

    def _get_grads(self):
        grads = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            #if p.grad is None:
            #    raise RuntimeError('Model parameter did not receive gradient: ' + name + '. '
                                                                                       #  'Use the param in the forward pass or set requires_grad=False')
            grads.append(p.grad.data)
        return grads

    def _get_flat_grads(self, out=None):
        grads = self._get_grads()
        if out is None:
            grads_size = sum(g.numel() for g in grads)
            out = grads[0].new(grads_size).zero_()
        offset = 0
        for g in grads:
            numel = g.numel()
            out[offset:offset+numel].copy_(g.contiguous().view(-1))
            offset += numel
        return out[:offset]

    def _set_flat_grads(self, new_grads):
        grads = self._get_grads()
        offset = 0
        for g in grads:
            numel = g.numel()
            g.copy_(new_grads[offset:offset+numel].view_as(g))
            offset += numel

    def _opt(self):
        # take an optimization step
        self.optimizer.step()
        self.zero_grad()
        self._num_updates += 1

        # update learning rate
        # self.lr_scheduler.step_update(self._num_updates)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def clear_buffered_stats(self):
        self._buffered_stats.clear()

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def _prepare_sample(self, sample):
        if sample is None or len(sample) == 0:
            return None
        return utils.move_to_cuda(sample)

    def dummy_train_step(self, dummy_batch):
        """Dummy training step for warming caching allocator."""
        self.train_step(dummy_batch, update_params=False)
        self.zero_grad()
        self.clear_buffered_stats()

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

