# -*- coding: utf-8 -*-
"""在缓存的共现矩阵上快速对比"生僻字抑制"等参数对思考内容的影响 (不用重读语料)。"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import CharBrain, Params, build_inhibition, build_weights  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PUNCT = set("，。、；：？！“”‘’（）《》〈〉【】—…·～,.!?;:\"'()[]")
SEEDS = ["苹果手机", "苹果香蕉", "苹果价格", "今天天气不错我想出去走走", "人工智能改变生活",
         "中国经济形势", "北京上海广州"]
CONFIGS = [
    ("先验关", dict(prior_k=0.0)),
    ("先验2000/0.35", dict(prior_k=2000.0, prior_pow=0.35)),
    ("先验4000/0.5", dict(prior_k=4000.0, prior_pow=0.5)),
    ("先验8000/0.6", dict(prior_k=8000.0, prior_pow=0.6)),
]


def load_counts(path):
    z = np.load(path, allow_pickle=False)
    V = len(z["chars"])
    C = np.zeros((V, V), dtype=np.int32)
    C[z["rows"], z["cols"]] = z["data"]
    return C, [str(c) for c in z["chars"]], z["freq"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default=os.path.join(HERE, "models", "counts_mid600m.npz"))
    ap.add_argument("--target", type=int, default=50)
    ap.add_argument("--soft", type=float, default=0.15)
    args = ap.parse_args()

    C, chars, freq = load_counts(args.counts)
    print(f"共现矩阵 V={len(chars)} 非零={int((C>0).sum()):,}")
    base = Params()
    base.target_active = args.target
    base.out_soft = args.soft
    base.decay = 0.35
    base.gain_control = 0.02
    base.steps = 20

    for name, kw in CONFIGS:
        pp = Params(**{**vars(base), **kw})
        W, Wn, st, factors = build_weights(C, pp, char_freq=freq)
        b = CharBrain(chars, W, Wn, build_inhibition(C, freq, pp), pp, {}, factors=factors)
        b.freq = freq
        print(f"\n===== {name} (col_typ={st['col_typ']:.2f}, 传播权重中位={st['prop_median']:.5f})")
        for s in SEEDS:
            tr = b.think(s, steps=20)
            content = [c for c, _ in tr.final if c not in PUNCT]
            ids = [b.char2id[c] for c in content if c in b.char2id]
            rare = float(np.mean([1.0 if freq[i] < 3000 else 0.0 for i in ids])) if ids else 0
            print(f"  [{s}] n={tr.n_mean:.0f} 生僻占比={rare:.2f} | " + "".join(content[:38]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
