# -*- coding: utf-8 -*-
"""最少机制比较: 到底哪些是必需的?

A. 现在: 增长曲线 + PMI 闸门 + 行列对称归一化 (α=0.5)
B. 去掉 PMI 闸门: 只有 增长曲线 + 归一化
C. 去掉 PMI 也去掉列归一化 (α=1, 即"给定当前字, 下一个字是谁"的条件概率)
D. 只有原始权重, 完全不归一化 (对照组)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import CharBrain, Params, build_inhibition, build_weights  # noqa: E402
from rebuild import load_counts  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PUNCT = set("，。、；：？！“”‘’（）《》〈〉【】—…·～,.!?;:\"'()[]")
FUNC = set("的是不我了和有为最在也就都而及与被把对从到着过得地吗呢吧啊这那你他它"
           "一二三四五六七八九十个人年月日时分上下前后中大多少好会很更还没只可以要")
SEEDS = ["苹果手机", "苹果香蕉", "股票市场价格", "医生给病人看病", "北京上海广州"]

VARIANTS = [
    ("A 增长+PMI+行列归一", dict(use_pmi=True, norm_alpha=0.5)),
    ("B 增长+行列归一 (无PMI)", dict(use_pmi=False, norm_alpha=0.5)),
    ("C 增长+行归一 (无PMI/无列)", dict(use_pmi=False, norm_alpha=1.0)),
]


def main() -> int:
    counts = os.path.join(HERE, "models", "counts_char_brain.npz")
    C, chars, freq = load_counts(counts)
    c2i = {c: i for i, c in enumerate(chars)}
    for tgt in (15, 20, 30):
        print(f"\n########## target_active={tgt}")
        for name, kw in VARIANTS:
            p = Params(prior_mode="off", conf_k=0.0, sharp_pow=1.0, **kw)
            p.target_active = tgt
            p.out_soft = 0.15
            p.decay = 0.35
            p.gain_control = 0.02
            p.steps = 20
            W, Wn, st, factors = build_weights(C, p, char_freq=freq)
            b = CharBrain(chars, W, Wn, build_inhibition(C, freq, p), p, {}, factors=factors)
            b.freq = freq
            print(f"\n== {name}")
            for s in SEEDS:
                tr = b.think(s, steps=20)
                content = [c for c, _ in tr.final if c not in PUNCT]
                ids = [c2i[c] for c in content if c in c2i]
                mf = float(np.mean([freq[i] for i in ids])) if ids else 0
                fn = sum(1 for c in content if c in FUNC)
                print(f"  [{s}] n={tr.n_mean:.0f} 字频={mf:,.0f} 虚词={fn} | "
                      + "".join(content[:30]))
        # 对照 D: 完全不归一化
        p = Params(prior_mode="off", conf_k=0.0, sharp_pow=1.0, use_pmi=False, norm_alpha=0.5)
        p.target_active = tgt
        p.out_soft = 0.15
        p.steps = 20
        W, Wn, st, factors = build_weights(C, p, char_freq=freq)
        b = CharBrain(chars, W, W.copy(), build_inhibition(C, freq, p), p, {}, factors=None)
        b.freq = freq
        print("\n== D 原始权重不归一化 (对照)")
        for s in SEEDS[:3]:
            tr = b.think(s, steps=20)
            content = [c for c, _ in tr.final if c not in PUNCT]
            print(f"  [{s}] n={tr.n_mean:.0f} | " + "".join(content[:30]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
