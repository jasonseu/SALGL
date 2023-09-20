# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-9-24
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
from os.path import join
import time
import yaml
import shutil
import random
import argparse
import logging
from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
from torch import optim

from .aslloss import AsymmetricLossOptimized


logger = logging.getLogger(__name__)
    

class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, comat_ema=False, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.comat_ema = comat_ema
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))
            if not self.comat_ema:
                self.module.comatrix = model.comatrix

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

    def forward(self, x):
        return self.module(x)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_trainable_params(model, cfg):
    if cfg.mode == 'full':
        group = model.parameters()
    elif cfg.mode == 'part':
        backbone, others = [], []
        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone.append(param)
            else:
                others.append(param)
        logger.info('backbone parameters: {} other parameters: {}'.format(len(backbone), len(others)))
        group = [
            {'params': backbone, 'lr': cfg.lr * 0.1},
            {'params': others, 'lr': cfg.lr}
        ]
    elif cfg.mode == 'layer4':
        if 'resnet' not in cfg.backbone.lower():
            raise Exception('layer4 training can only be used in ResNet fanmily models!')
        for name, param in model.backbone.named_parameters():
            param.requires_grad = False
            if 'layer4' in name:
                param.requires_grad = True
        group = list(filter(lambda p: p.requires_grad, model.parameters()))
    return group

def get_optimizer(params, cfg):
    if cfg.optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adam':
        optimizer = optim.Adam(
            params, 
            lr=cfg.lr, 
            betas=(0.9, 0.999), 
            eps=1e-08, 
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            params, 
            lr=cfg.lr, 
            betas=(0.9, 0.999), 
            eps=1e-08, 
            weight_decay=cfg.weight_decay
        )
    return optimizer

def get_lr_scheduler(optimizer, cfg, steps_per_epoch=0):
    if cfg.lr_scheduler == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1, verbose=True)
    elif cfg.lr_scheduler == 'StepLR':
        return optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=0.1)
    elif cfg.lr_scheduler == 'OneCycleLR':
        return optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=cfg.lr, 
            steps_per_epoch=steps_per_epoch,
            epochs=cfg.max_epochs,
            pct_start=cfg.pct_start
        )
    else:
        raise Exception('lr scheduler {} not found!'.format(cfg.lr_scheduler))

def get_loss_fn(cfg):
    if cfg.loss == 'bce':
        return nn.BCEWithLogitsLoss()
    elif cfg.loss == 'asl':
        # return AsymmetricLoss(cfg.gamma_neg, cfg.gamma_pos, cfg.clip, disable_torch_grad_focal_loss=True)
        return AsymmetricLossOptimized(
            gamma_neg=cfg.gamma_neg,
            gamma_pos=cfg.gamma_pos,
            clip=cfg.clip,
            disable_torch_grad_focal_loss=True,
            eps=1e-05
        )
    else:
        raise Exception('loss function {} not found!'.format(cfg.loss))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def get_experiment_id(exp_home):
    exp_names = [t for t in os.listdir(exp_home) if t[-1].isdigit()]
    if len(exp_names) == 0:
        new_exp_id = 1
    else:
        exp_ids = [int(en[3:]) for en in exp_names]
        new_exp_id = max(exp_ids) + 1
    return new_exp_id

def check_makedir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        
def check_exists(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError('file {} not found!'.format(filepath))

def prepare_env(args, argv):
    # prepare data config
    cfg = vars(args)
    cfg['train_path'] = join('data', args.data, 'train.txt')
    cfg['test_path'] = join('data', args.data, 'test.txt')
    cfg['label_path'] = join('data', args.data, 'label.txt')
    cfg['embed_path'] = join('data', args.data, '{}.npy'.format(cfg['embed_type']))
    cfg['comat_path'] = join('data', args.data, 'comatrix.npy')
    cfg['ignore_path'] = join('data', args.data, 'ignore.npy')
    check_exists(cfg['train_path'])
    check_exists(cfg['test_path'])
    check_exists(cfg['label_path'])
    cfg['num_classes'] = len(open(cfg['label_path']).readlines())
    
    # prepare checkpoint and log config
    exp_home = join('experiments', '{}_{}_{}'.format(args.model, args.backbone, args.data))
    check_makedir(exp_home)
    exp_name = 'exp{}'.format(get_experiment_id(exp_home))
    exp_dir = join(exp_home, exp_name)
    cfg['exp_dir'] = exp_dir
    cfg['log_path'] = join(exp_dir, 'train.log')
    cfg['ckpt_dir'] = join(exp_dir, 'checkpoints')
    cfg['ckpt_best_path'] = join(cfg['ckpt_dir'], 'best_model.pth')
    cfg['ckpt_ema_best_path'] = join(cfg['ckpt_dir'], 'ema_best_model.pth')
    cfg['ckpt_latest_path'] = join(cfg['ckpt_dir'], 'latest_model.pth')
    check_makedir(cfg['exp_dir'])
    check_makedir(cfg['ckpt_dir'])
    
    # save experiment checkpoint
    exp_ckpt_path = os.path.join(exp_home, 'checkpoint.txt')
    temp = ' '.join(['python', *argv])
    with open(exp_ckpt_path, 'a') as fa:
        fa.writelines('{}\t{}\n'.format(exp_name, temp))
    
    # save config
    cfg_path = join(cfg['exp_dir'], 'config.yaml')
    with open(cfg_path, 'w') as fw:
        for k, v in cfg.items():
            fw.write('{}: {}\n'.format(k, v))
            
    cfg = argparse.Namespace(**cfg)
    log_path = join(exp_dir, 'train.log')
    prepare_log(log_path, cfg)
            
    return cfg

def prepare_log(log_path, cfg, level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(level)
    sh = logging.StreamHandler()
    th = logging.FileHandler(filename=log_path, encoding='utf-8')
    logger.addHandler(sh)
    logger.addHandler(th)
    
    logger.info('model training time: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    logger.info('model configuration: ')
    format_string = cfg.__class__.__name__ + '(\n'
    for k, v in vars(cfg).items():
        format_string += '    {}: {}\n'.format(k, v)
    format_string += ')'
    logger.info(format_string)
    
def get_logger(log_path, name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    steam_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=log_path, encoding='utf-8')
    logger.addHandler(steam_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger
    
def clear_exp(exp_dir):
    logging.shutdown()
    shutil.rmtree(exp_dir)
    exp_home = os.path.dirname(exp_dir)
    exp_ckpt_path = os.path.join(exp_home, 'checkpoint.txt')
    with open(exp_ckpt_path, 'r') as fr:
        temp = fr.readlines()[:-1]
    with open(exp_ckpt_path, 'w') as fw:
        fw.writelines(temp)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='salgl')
    parser.add_argument('--backbone', type=str, default='resnet101')
    parser.add_argument('--data', type=str, default='voc2007')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--lr-scheduler', type=str, default='OneCycleLR')
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--loss', type=str, default='asl')
    parser.add_argument('--gamma-pos', type=float, default=0.0)
    parser.add_argument('--gamma-neg', type=float, default=2.0)
    parser.add_argument('--clip', type=float, default=0.00)
    parser.add_argument('--embed-type', type=str, default='bert', choices=['glove', 'bert', 'random'])
    parser.add_argument('--ema-decay', type=float, default=0.9997)
    parser.add_argument('--outmess', action='store_true')
    parser.add_argument('--no-comat-ema', dest='comat_ema', action='store_false')
    parser.add_argument('--bb-eval', action='store_true')
    parser.add_argument('--orid-norm', action='store_true')
    parser.add_argument('--pos', action='store_true')
    
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--img-size', type=int, default=448)
    parser.add_argument('--num-scenes', type=int, default=4)
    parser.add_argument('--num-steps', type=int, default=3)
    parser.add_argument('--mode', type=str, default='full')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--ignore-self', action='store_true')
    parser.add_argument('--soft', action='store_true')
    parser.add_argument('--lamda', type=float, default=1.0)
    parser.add_argument('--pct-start', type=float, default=0.2)
    parser.add_argument('--estop', action='store_true')
    parser.add_argument('--cnt-clip', type=float, default=0.0)
    
    parser.add_argument('--max-epochs', type=int, default=80)
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--pretrained-model', type=str, default=None)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()
    return args
