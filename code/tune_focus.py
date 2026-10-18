# -*- coding: utf-8 -*-
"""只靠"调参"让思考聚焦: 扫激活规模 target_active (及其它动力学参数), 不加新机制。

权重侧一律用最朴素的配置:
    prior_mode=off     (不压生僻字)
    conf_k=0           (不按来源可靠度打折)
    sharp_pow=1.0      (不锐化, 直接用连接强度)
    norm_alpha=0.5     (只保留"出边摊薄 + 入边摊薄"这一个归一化)
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import CharBrain, Params, build_inhibition, build_weights  # noqa: E402
from rebuild import load_counts  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PUNCT = set("，。、；：？！“”‘’（）《》〈〉【】—…·～,.!?;:\"'()[]")
FUNC = set("的是不我了和有为最在也就都而及与被把对从到着过得地吗呢吧啊这那你他它"
           "一二三四五六七八九十个年月日时分上下前后中大多少好会很更还没只可以要")
SEEDS = ["苹果手机", "苹果香蕉", "北京上海广州", "今天天气不错我想出去走走", "人工智能技术",
         "医生给病人看病", "股票市场价格"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default=os.path.join(HERE, "models", "counts_char_brain.npz"))
    ap.add_argument("--weights", default="plain", choices=["plain", "current"],
                    help="plain=最朴素权重(只保留归一化) current=出厂设置")
    args = ap.parse_args()

    C, chars, freq = load_counts(args.counts)
    c2i = {c: i for i, c in enumerate(chars)}

    if args.weights == "plain":
        wp = dict(prior_mode="off", conf_k=0.0, sharp_pow=1.0, norm_alpha=0.5,
                  use_pmi=True, k_sat=3.0, c_ref=3000.0, top_k=64, c_min=3)
    else:
        wp = {}
    p0 = Params(**wp)
    W, Wn, st, factors = build_weights(C, p0, char_freq=freq)
    inh = build_inhibition(C, freq, p0)
    print(f"权重({args.weights}): 连接 {st['edges_kept']:,}, col_typ={st['col_typ']:.2f}, "
          f"传播权重中位 {st['prop_median']:.5f}")

    b = CharBrain(chars, W, Wn, inh, p0, {}, factors=factors)
    b.freq = freq

    print(f"\n{'target':>7} {'soft':>6} {'decay':>6} | 输出 (前 32 字)")
    for tgt, soft, decay in itertools.product([10, 15, 20, 30, 50], [0.15], [0.35]):
        p0.target_active = tgt
        p0.out_soft = soft
        p0.decay = decay
        p0.gain_control = 0.02
        p0.steps = 20
        print(f"\n--- target_active={tgt}")
        for s in SEEDS:
            tr = b.think(s, steps=20)
            content = [c for c, _ in tr.final if c not in PUNCT]
            ids = [c2i[c] for c in content if c in c2i]
            mf = float(np.mean([freq[i] for i in ids])) if ids else 0
            fn = sum(1 for c in content if c in FUNC)
            print(f"  [{s}] n={tr.n_mean:.0f} 平均字频={mf:,.0f} 虚词={fn} | "
                  + "".join(content[:32]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
