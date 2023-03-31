import math
import re
from collections import defaultdict
import torch
from torch.optim.optimizer import Optimizer


class AdamSVD(Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-2, betas=(0.9, 0.999), eps=1e-8, svd=True, thres=600.,
                 weight_decay=0, amsgrad=False, ratio=0.8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, svd=svd, thres=thres)
        super(AdamSVD, self).__init__(params, defaults)

        self.eigens = defaultdict(dict)
        self.transforms = defaultdict(dict)
        self.ratio = ratio

    def __setstate__(self, state):
        super(AdamSVD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('svd', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            svd = group['svd']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamSVD does not support sparse gradients, please consider SparseAdam instead')

                update = self.get_update(group, grad, p)
                if svd and len(self.transforms) > 0:
                    # if len(update.shape) == 4:
                    if len(update.shape) == 3:
                        # the transpose of the manuscript
                        update_ = torch.bmm(update, self.transforms[p]).view_as(update)

                    else:
                        update_ = torch.mm(update, self.transforms[p])

                else:
                    update_ = update
                p.data.add_(update_)
        return loss

    def get_transforms(self):
        for group in self.param_groups:
            svd = group['svd']
            if svd is False:
                continue
            for p in group['params']:
                if p.grad is None:
                    continue
                thres = group['thres']
                temp = []
                for s in range(self.eigens[p]['eigen_value'].shape[0]):
                    ind = self.eigens[p]['eigen_value'][s] <= self.eigens[p]['eigen_value'][s][-1] * thres
                    ind = torch.ones_like(ind)
                    ind[: int(ind.shape[0]*(1.0-self.ratio))] = False
                    # GVV^T
                    # get the columns
                    basis = self.eigens[p]['eigen_vector'][s][:, ind]
                    transform = torch.mm(basis, basis.transpose(1, 0))
                    temp.append(transform/torch.norm(transform))
                self.transforms[p] = torch.stack(temp, dim=0)
                self.transforms[p].detach_()

    def get_eigens(self, fea_in):
        for group in self.param_groups:
            svd = group['svd']
            if svd is False:
                continue
            for idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                eigen = self.eigens[p]
                eigen_values, eigen_vectors = [], []
                for s in range(fea_in[idx].shape[0]):
                    _, eigen_value, eigen_vector = torch.svd(fea_in[idx][s], some=False)
                    eigen_values.append(eigen_value)
                    eigen_vectors.append(eigen_vector)
                eigen['eigen_value'] = torch.stack(eigen_values, dim=0)
                eigen['eigen_vector'] = torch.stack(eigen_vectors, dim=0)

    def get_update(self, group, grad, p):
        amsgrad = group['amsgrad']
        state = self.state[p]

        # State initialization
        if len(state) == 0:
            state['step'] = 0
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p.data)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p.data)
            if amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state['max_exp_avg_sq'] = torch.zeros_like(p.data)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        if amsgrad:
            max_exp_avg_sq = state['max_exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1

        if group['weight_decay'] != 0:
            grad.add_(group['weight_decay'], p.data)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = max_exp_avg_sq.sqrt().add_(group['eps'])
        else:
            denom = exp_avg_sq.sqrt().add_(group['eps'])

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * \
            math.sqrt(bias_correction2) / bias_correction1
        update = - step_size * exp_avg / denom
        return update


def init_model_optimizer(self):
    fea_params = [p for n, p in self.model.named_parameters(
    ) if not bool(re.match('last', n)) and 'bn' not in n]
    bn_params = [p for n, p in self.model.named_parameters() if 'bn' in n]
    model_optimizer_arg = {'params': [{'params': fea_params, 'svd': True, 'lr': self.svd_lr,
                                        'thres': self.config['svd_thres']},
                                      {'params': bn_params, 'lr': self.config['bn_lr']}],
                           'lr': self.config['model_lr'],
                           'weight_decay': self.config['model_weight_decay']}
    if self.config['model_optimizer'] in ['SGD', 'RMSprop']:
        model_optimizer_arg['momentum'] = self.config['momentum']
    elif self.config['model_optimizer'] in ['Rprop']:
        model_optimizer_arg.pop('weight_decay')
    elif self.config['model_optimizer'] in ['amsgrad']:
        if self.config['model_optimizer'] == 'amsgrad':
            model_optimizer_arg['amsgrad'] = True
        self.config['model_optimizer'] = 'Adam'

    self.model_optimizer = AdamSVD(**model_optimizer_arg)
    self.model_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.model_optimizer,
                                                                milestones=self.config['schedule'],
                                                                gamma=self.config['gamma'])

