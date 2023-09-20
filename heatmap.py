# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-8-9
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import yaml
import warnings
import argparse
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import torch

warnings.filterwarnings("ignore")
labelmap = {
    'motorcycle': 'mcycle',
    'traffic light': 'tlight',
    'stop sign': 'ssign',
    'potted plant': 'pplant',
    'dining table': 'dtable',
    'keyboard': 'kboard',
    'parking meter': 'pmeter'
}


select_labels = [
    [1,  2,  3,  5,  0,  6,  7,  9, 11, 12],
    [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    [57, 58, 59, 60, 0, 56, 62, 63, 64, 66],
    [39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
]


def comat2prob(comat):
    comat = torch.from_numpy(comat)
    temp = torch.diagonal(comat, dim1=1, dim2=2).unsqueeze(-1) 
    comat = comat / (temp + 1e-8)  # divide diagonal

    mask = torch.eye(comat.shape[-1]).unsqueeze(0)
    masks = torch.cat([mask for _ in range(comat.shape[0])], dim=0)
    comat = comat * (1 - masks)

    comat = comat.numpy()
    return comat


def cooccur_prob_visualization(prob, save_dir, name, xlabels, ylabels):
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticklabels(xlabels, rotation=45, ha='right')
    ax.set_xticks(range(len(xlabels)))
    ax.set_yticklabels(ylabels)
    ax.set_yticks(range(len(ylabels)))
    plt.imshow(prob, cmap=sns.cm.rocket)
    cb = plt.colorbar()
    cb.outline.set_visible(False)
    plt.clim(0, 1)
    dx = 8 / 72.
    dy = 3 / 72.
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
    plt.savefig(os.path.join(save_dir, name), dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()


def main(format, cfg):
    save_dir = 'heatmap'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    labels = [line.strip() for line in open(cfg.label_path)]
    ckpt = torch.load(cfg.ckpt_ema_best_path, map_location='cpu')
    comat = ckpt['module.comatrix'].numpy()
    prob = comat2prob(comat)
    for i in range(comat.shape[0]):
        for j in range(len(select_labels)):
            if (i == 0 or i == 2) and (j == 0 or j == 2):
                temp = select_labels[j]
                _labels = []
                for t in temp:
                    if labels[t] in labelmap.keys():
                        _labels.append(labelmap[labels[t]])
                    else:
                        _labels.append(labels[t])
                _comat = comat[i]
                _comat = _comat[temp][:, temp]
                np.fill_diagonal(_comat, 0)
                _prob = prob[i]
                _prob = _prob[temp][:, temp]
                name = '{}{}.{}'.format(i, j, format)
                cooccur_prob_visualization(_prob, save_dir, name, _labels, _labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', type=str, default='experiments/salgl_resnet101_mscoco/exp3')
    parser.add_argument('--format', type=str, default='jpg', choices=['jpg', 'pdf'])
    args = parser.parse_args()
    cfg_path = os.path.join(args.exp_dir, 'config.yaml')
    cfg = yaml.load(open(cfg_path, 'r'))
    cfg = argparse.Namespace(**cfg)
    main(args.format, cfg)
