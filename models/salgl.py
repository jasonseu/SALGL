# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-7-15
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .utils import *
from lib.utils import concat_all_gather
from .factory import create_backbone, register_model


class EntropyLoss(nn.Module):
    def __init__(self, margin=0.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, x, eps=1e-7):
        x = x * torch.log(x + eps)
        en = -1 * torch.sum(x, dim=-1)
        en = torch.mean(en)
        return en


class SALGL(nn.Module):
    """Ablation study. SALGL model without the fisrt stage graph propagation for label embedding."""
    def __init__(self, backbone, feat_dim, cfg, att_dim=1024):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.tb = TransformerEncoder(d_model=feat_dim, nhead=8, num_layers=1)
        if cfg.pos:
            self.position_embedding = build_position_encoding(feat_dim, cfg.backbone, 'sine', cfg.img_size)
        if self.cfg.embed_type == 'random':
            self.embeddings = nn.Parameter(torch.empty((cfg.num_classes, feat_dim)))
            nn.init.normal_(self.embeddings)
        else:
            self.register_buffer('embeddings', torch.from_numpy(np.load(cfg.embed_path)).float())
        embed_dim = self.embeddings.shape[-1]
        self.entropy = EntropyLoss()
        self.max_en = self.entropy(torch.tensor([1 / cfg.num_scenes] * cfg.num_scenes).cuda())

        self.scene_linear = nn.Linear(feat_dim, cfg.num_scenes, bias=False)
        self.ggnn = GatedGNN(feat_dim, cfg)
        self.register_buffer('comatrix', torch.zeros((cfg.num_scenes, cfg.num_classes, cfg.num_classes)))

        self.attention = LowRankBilinearAttention(feat_dim, embed_dim, att_dim)
        _feat_dim = feat_dim * 2 if cfg.num_steps > 0 else feat_dim
        self.fc = nn.Sequential(
            nn.Linear(_feat_dim, feat_dim),
            nn.Tanh()
        )
        self.classifier = Element_Wise_Layer(cfg.num_classes, feat_dim)

    def comatrix2prob(self):
        # comat: [bs, nc, nc]
        comat = torch.transpose(self.comatrix, dim0=1, dim1=2)
        temp = torch.diagonal(comat, dim1=1, dim2=2).unsqueeze(-1)  # [ns, nc, 1]
        comat = comat / (temp + 1e-8)  # divide diagonal

        if self.cfg.ignore_self:  # make the diagonal zeros
            mask = torch.eye(self.cfg.num_classes).cuda().unsqueeze(0)  # [1, nc, nc]
            masks = torch.cat([mask for _ in range(comat.shape[0])], dim=0)  # [ns, nc, nc]
            comat = comat * (1 - masks)

        if self.cfg.normalize:  # divide summation
            temp = torch.sum(comat, dim=-1).unsqueeze(-1)  # [ns, nc, 1]
            comat = comat / (temp + 1e-8)

        return comat

    def forward(self, x, y=None):
        if self.training and self.cfg.soft:
            self.comatrix.detach_()

        img_feats = self.backbone(x)
        pos = None
        if self.cfg.pos:
            pos = self.position_embedding(x)
            pos = torch.flatten(pos, 2).transpose(1, 2)
        img_feats = self.tb(img_feats, pos=pos)

        img_contexts = torch.mean(img_feats, dim=1)
        scene_scores = self.scene_linear(img_contexts)

        if self.training:
            _scene_scores = scene_scores
            _scene_probs = F.softmax(_scene_scores, dim=-1)
            batch_comats = torch.bmm(y.unsqueeze(2), y.unsqueeze(1))  # [bs, nc, nc]
            if self.cfg.distributed:
                _scene_probs = concat_all_gather(_scene_probs)
                batch_comats = concat_all_gather(batch_comats)
            for i in range(_scene_probs.shape[0]):
                if self.cfg.soft:
                    prob = _scene_probs[i].unsqueeze(-1).unsqueeze(-1)  # [num_scenes, 1, 1]
                    comat = batch_comats[i].unsqueeze(0)  # [1, num_classes, num_clasees]
                    self.comatrix += prob * comat
                else:
                    maxsid = torch.argmax(_scene_probs[i])
                    self.comatrix[maxsid] += batch_comats[i]

        scene_probs = F.softmax(scene_scores, dim=-1)

        sample_en = self.entropy(scene_probs)
        _scene_probs = torch.mean(scene_probs, dim=0)
        batch_en = (self.max_en - self.entropy(_scene_probs)) * 100

        # label vector embedding
        label_feats = self.embeddings.unsqueeze(0).repeat(x.shape[0], 1, 1)

        # compute visual representation of label
        label_feats, alphas = self.attention(img_feats, label_feats)

        # graph propagation
        if not self.cfg.soft:
            comats = self.comatrix2prob()
            indices = torch.argmax(scene_probs, dim=-1)
            comats = torch.index_select(comats, dim=0, index=indices)
        else:
            _scene_probs = scene_probs.unsqueeze(-1).unsqueeze(-1)  # [bs, num_scenes, 1, 1]
            comats = self.comatrix2prob().unsqueeze(0)  # [1, num_scenes, nc, nc]
            comats = _scene_probs * comats  # [bs, num_scenes, nc, nc]
            comats = torch.sum(comats, dim=1)  # [bs nc, nc]

        output = self.ggnn(label_feats, comats)

        if self.cfg.num_steps > 0:
            output = torch.cat([label_feats, output], dim=-1)
        output = self.fc(output)
        logits = self.classifier(output)

        return {
            'logits': logits,
            'scene_probs': scene_probs,
            'sample_en': sample_en,
            'batch_en': batch_en,
            'att_weights': alphas,
            'comat': comats
        }


@register_model
def salgl(pretrained, cfg):
    backbone, feat_dim = create_backbone(cfg.backbone, pretrained, img_size=cfg.img_size)
    model = SALGL(backbone, feat_dim, cfg)
    return model
