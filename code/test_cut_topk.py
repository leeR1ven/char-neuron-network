# -*- coding: utf-8 -*-
"""拐点 (cut_c) 与出边数 (top_k) 的权衡:
   拐点太低 -> 手机/机器/医院 这类高频真词被压低, 再被 top_k 剪掉 (网络里没了)
   拐点太高 -> 的→是 这类通用搭配又回来了
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import CharBrain, Params, build_inhibition, build_weights  # noqa: E402
from rebuild import load_counts  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK = [("手", "机"), ("机", "器"), ("医", "院"), ("苹", "果"), ("月", "亮"),
         ("芯", "片"), ("的", "是"), ("的", "了"), ("一", "个"), ("我", "们")]

C, chars, freq = load_counts(os.path.join(HERE, "models", "counts_char_brain.npz"))
order = np.argsort(-freq)
generic = set(order[:200].tolist())
print(f"V={len(chars)}\n")
print(f"{'cut':>9} {'topk':>5} | 关键搭配保留情况 | 通用字出边均值/全网 | 边数")
for cut in (1e9, 1e6, 3e5, 1e5):
    for topk in (32, 64):
        p = Params(growth="hill", rise_k=30.0, rise_a=0.5, cut_c=cut, cut_b=1.0,
                   use_norm=False, top_k=topk)
        W, Wn, st, fac = build_weights(C, p, char_freq=freq)
        keep = []
        for a, b in CHECK:
            i, j = chars.index(a), chars.index(b)
            w = W[i, j]
            keep.append(f"{a}{b}:{w:.2f}" if w > 0 else f"{a}{b}:×")
        gen = float(np.mean([W[i][W[i] > 0].mean() for i in generic]))
        allw = float(W[W > 0].mean())
        print(f"{cut:9.0f} {topk:5d} | " + " ".join(f"{k:>8s}" for k in keep)
              + f" | {gen:.3f}/{allw:.3f}={gen/allw:.2f} | {st['edges_kept']:,}")
