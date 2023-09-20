# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2023-9-20
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2023 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import yaml
import torch
import argparse
import numpy as np
import seaborn as sns
from wordcloud import WordCloud


sns.set_theme(style="whitegrid")


def main(cfg):
    ckpt_path = os.path.join(cfg.exp_dir, 'checkpoints/ema_best_model.pth')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    comatrix = ckpt['module.comatrix'].numpy()
    diags = np.diagonal(comatrix, axis1=1, axis2=2)
    labels = [line.strip() for line in open(os.path.join('data', cfg.data, 'label.txt'))]

    save_dir = './wordcloud'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    temp = diags.sum(axis=0)
    for i in range(diags.shape[0]):
        d = {}
        for j, (w, c) in enumerate(zip(labels, diags[i].tolist())):
            d[w] = int(c / temp[j] * 200)

        wordcloud = WordCloud(background_color="white")
        wordcloud.generate_from_frequencies(frequencies=d)
        wordcloud.to_file(os.path.join(save_dir, 'wordcloud_{}.png'.format(i)))
        svg = wordcloud.to_svg(embed_font=True)
        save_path = os.path.join(save_dir, 'wordcloud_{}.svg'.format(i))
        with open(save_path, "wb") as f:
            f.write(svg.encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-dir', type=str, default='experiments/salgl_resnet101_mscoco/exp3')
    args = parser.parse_args()
    cfg_path = os.path.join(args.exp_dir, 'config.yaml')
    cfg = yaml.load(open(cfg_path, 'r'))
    cfg = argparse.Namespace(**cfg)
    main(cfg)
