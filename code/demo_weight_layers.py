# -*- coding: utf-8 -*-
"""对比: 直接用"连接强度 W"传播  vs  用"传播权重 Ŵ"传播。

证明为什么不能只保留一个权重: 直接用 W 时, 出边多/入边多的常见字会瞬间淹没思考。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import CharBrain  # noqa: E402

PUNCT = set("，。、；：？！“”‘’（）《》〈〉【】—…·～,.!?;:\"'()[]")
SEEDS = ["苹果手机", "苹果香蕉", "北京上海", "人工智能技术"]


def show(tag, b):
    print(f"\n===== {tag}")
    for s in SEEDS:
        tr = b.think(s, steps=20)
        content = [c for c, _ in tr.final if c not in PUNCT]
        ids = [b.char2id[c] for c in content if c in b.char2id]
        mean_f = float(np.mean([b.freq[i] for i in ids])) if ids else 0
        funcs = sum(1 for c in content if c in "的是不我了和有为在也就都而及与被把对从")
        print(f"  [{s}] n={tr.n_mean:.0f} 平均字频={mean_f:,.0f} 虚词={funcs}/{len(content)}")
        print("      " + "".join(content[:36]))


path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "char_brain.npz")
b = CharBrain.load(path)

# A: 现状 (连接强度 W -> 传播权重 Ŵ)
show("A. 用传播权重 Ŵ (现行)", b)

# B: 只有一种权重 —— 直接拿连接强度 W 传播
b2 = CharBrain.load(path)
b2.Wn = b2.W.copy()
show("B. 直接用连接强度 W 传播 (只有一种权重)", b2)

# C: 只做行归一化 (最常见的"条件概率"做法)
b3 = CharBrain.load(path)
rs = b3.W.sum(axis=1, keepdims=True)
b3.Wn = b3.W / np.maximum(rs, 1e-9)
show("C. 只做行归一化 (P(下一个字|当前字))", b3)
