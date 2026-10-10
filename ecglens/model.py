"""Small 1D-ResNet ECG encoder + text projection head + multi-positive InfoNCE."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1, k=5):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride, k // 2, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, 1, k // 2, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.skip is None else self.skip(x)))


class ECGEncoder(nn.Module):
    """12-lead, 100 Hz, 10 s (12 x 1000) -> 128-d L2-normalised embedding. ~1.2M params."""

    def __init__(self, emb_dim=128, widths=(32, 64, 128, 192)):
        super().__init__()
        w = widths
        self.stem = nn.Sequential(nn.Conv1d(12, w[0], 15, 2, 7, bias=False), nn.BatchNorm1d(w[0]), nn.ReLU())
        self.stages = nn.Sequential(
            Block(w[0], w[0], 2), Block(w[0], w[0]),
            Block(w[0], w[1], 2), Block(w[1], w[1]),
            Block(w[1], w[2], 2), Block(w[2], w[2]),
            Block(w[2], w[3], 2), Block(w[3], w[3]),
        )
        self.feat_dim = 2 * w[3]
        self.proj = nn.Linear(self.feat_dim, emb_dim)

    def features(self, x):
        h = self.stages(self.stem(x))
        return torch.cat([h.mean(-1), h.amax(-1)], dim=1)  # avg+max pooled backbone features (384-d)

    def forward(self, x):
        return F.normalize(self.proj(self.features(x)), dim=-1)


class TextHead(nn.Module):
    """Trainable MLP on top of frozen MiniLM sentence embeddings (384-d)."""

    def __init__(self, in_dim=384, hid=256, emb_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.GELU(), nn.Linear(hid, emb_dim))

    def forward(self, t):
        return F.normalize(self.net(t), dim=-1)


class ECGLens(nn.Module):
    def __init__(self):
        super().__init__()
        self.ecg = ECGEncoder()
        self.text = TextHead()
        self.log_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))


def multi_positive_infonce(ze, zt, sid, log_scale):
    """Symmetric InfoNCE where every in-batch pair with the SAME report string is a positive.

    PTB-XL reports are heavily templated ('sinus rhythm. normal ecg.' = 8% of rows), so plain InfoNCE would
    push apart pairs whose reports are literally identical. The target is uniform over all same-string items.
    """
    logits = log_scale.clamp(max=math.log(100)).exp() * ze @ zt.T
    pos = (sid[:, None] == sid[None, :]).float()
    tgt_e2t = pos / pos.sum(1, keepdim=True)
    tgt_t2e = pos.T / pos.T.sum(1, keepdim=True)
    l1 = -(tgt_e2t * F.log_softmax(logits, dim=1)).sum(1).mean()
    l2 = -(tgt_t2e * F.log_softmax(logits.T, dim=1)).sum(1).mean()
    return 0.5 * (l1 + l2)
