# -*- coding: utf-8 -*-
"""思考参数微调: 直接打印不同参数下的"思考内容", 用眼睛判断发散/聚焦。

关注: 激活规模 target_active、输出软度 out_soft、衰减 decay、增益控制 gain_control、
      驱动方式 drive_mode(sum/lse)、种子强度 seed_gain。
指标: 激活字的平均字频 (太低说明全是生僻字)、生僻字占比、虚词占比。
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import CharBrain  # noqa: E402

FUNC = set("的是不我了和有为最在也就都而及与被把对从到着过得地吗呢吧啊这那你他它"
           "一二三四五六七八九十个年月日时分上下前后中大多少好会很更还没只可以要"
           "，。、；：？！“”‘’（）《》—…·～")
PUNCT = set("，。、；：？！“”‘’（）《》《》〈〉【】—…·～,.!?;:\"'()[]")

SEEDS = ["苹果手机", "苹果香蕉", "今天天气不错我想出去走走", "人工智能改变生活"]


def summarize(brain: CharBrain, seeds: str, steps: int = 20):
    tr = brain.think(seeds, steps=steps)
    freq = getattr(brain, "freq", None)
    content = [c for c, _ in tr.final if c not in PUNCT]
    if freq is None:
        return tr, content, float("nan"), float("nan")
    ids = [brain.char2id[c] for c in content if c in brain.char2id]
    mf = float(np.mean([freq[i] for i in ids])) if ids else 0.0
    rare = float(np.mean([1.0 if freq[i] < 100 else 0.0 for i in ids])) if ids else 0.0
    return tr, content, mf, rare


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "models", "mid600m.npz"))
    ap.add_argument("--grid", default="target,soft,decay,gain,mode")
    args = ap.parse_args()
    b = CharBrain.load(args.model)
    p = b.p
    print(f"模型 {args.model}: V={b.V}")

    targets = [15, 25, 40, 60]
    softs = [0.1, 0.25]
    decays = [0.25, 0.45]
    gains = [0.01, 0.03]
    modes = ["sum", "lse"]
    rows = []
    for tgt, soft, decay, gain, mode in itertools.product(targets, softs, decays, gains, modes):
        p.target_active = tgt
        p.out_soft = soft
        p.decay = decay
        p.gain_control = gain
        p.drive_mode = mode
        p.lse_beta = 12.0
        p.steps = 20
        recs = []
        for s in SEEDS:
            tr, content, mf, rare = summarize(b, s, steps=20)
            recs.append((s, tr.n_mean, mf, rare, content))
        avg_rare = float(np.mean([r[3] for r in recs if not np.isnan(r[3])]))
        avg_freq = float(np.nanmean([r[2] for r in recs]))
        rows.append((avg_rare, avg_freq, (tgt, soft, decay, gain, mode), recs))

    rows.sort(key=lambda r: (r[0], -r[1]))
    print(f"\n{'生僻字占比':>10} {'平均字频':>10}  参数")
    for rare, mf, cfg, _ in rows[:10]:
        print(f"{rare:10.2f} {mf:10.0f}  target={cfg[0]} soft={cfg[1]} decay={cfg[2]} "
              f"gain={cfg[3]} mode={cfg[4]}")
    for rare, mf, cfg, recs in rows[:3]:
        print(f"\n=== target={cfg[0]} soft={cfg[1]} decay={cfg[2]} gain={cfg[3]} mode={cfg[4]} "
              f"(生僻字{rare:.2f}, 平均字频{mf:.0f})")
        for s, n, _, _, content in recs:
            print(f"  [{s}] n={n:.0f}  " + "".join(content[:40]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
