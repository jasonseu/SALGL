# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-8-9
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import yaml
import argparse
from argparse import Namespace
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.factory import create_model
from lib.dataset import MLDataset
from lib.utils import ModelEma
from lib.metrics import VOCmAP


torch.backends.cudnn.benchmark = True


class Evaluator(object):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        dataset = MLDataset(cfg.test_path, cfg, training=False)
        self.dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
        self.labels = dataset.labels

        self.model = create_model(cfg.model, cfg.pretrained, cfg=cfg)
        self.model.cuda()
        self.model = ModelEma(self.model, decay=cfg.ema_decay, comat_ema=cfg.comat_ema)

        self.cfg = cfg
        self.voc12_mAP = VOCmAP(cfg.num_classes, year='2012', ignore_path=cfg.ignore_path)

    @torch.no_grad()
    def run(self):
        model_dict = torch.load(self.cfg.ckpt_ema_best_path)
        self.model.load_state_dict(model_dict)
        print(f'loading best checkpoint success')

        self.model.eval()
        self.voc12_mAP.reset()
        cnt = np.zeros(self.cfg.num_scenes)
        for batch in tqdm(self.dataloader):
            img = batch['img'].cuda()
            targets = batch['target'].numpy()
            ret = self.model(img)
            logit = ret['logits']
            scores = torch.sigmoid(logit).cpu().numpy()
            self.voc12_mAP.update(scores, targets)

            scene_probs = ret['scene_probs'].cpu().numpy()
            for i in np.argmax(scene_probs, axis=1):
                cnt[i] += 1

        probs = cnt / np.sum(cnt) + 1e-8
        entropy = -np.sum(probs * np.log(probs))
        print('Scene category distribution entropy: {:.3f}'.format(entropy))

        _, mAP = self.voc12_mAP.compute()
        print('model {} data {} mAP: {:.3f}'.format(self.cfg.model, self.cfg.data, mAP))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', type=str, default='experiments/salgl_resnet101_mscoco/exp3')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--exp')
    args = parser.parse_args()
    cfg_path = os.path.join(args.exp_dir, 'config.yaml')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError('config file not found in the {}!'.format(cfg_path))
    cfg = yaml.load(open(cfg_path, 'r'))
    cfg = Namespace(**cfg)
    cfg.batch_size = args.batch_size
    cfg.threshold = args.threshold
    print(cfg)

    evaluator = Evaluator(cfg)
    evaluator.run()
