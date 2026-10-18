# -*- coding: utf-8 -*-
"""消融实验: "生僻目标抑制"到底压掉了噪声, 还是压死了生僻字?

三组对照 (都在同一份共现矩阵上重建权重):
  off    : 不压
  sat    : 全频段平滑压制 (K=8000, pow=0.6)  ← 之前用的
  tail   : 只压冷僻尾巴 (字频 >= 3000 完全不罚)

每题都测四个量:
  噪声率    : 随机常用种子激活出来的字里, 字频 < 3000 的占比 (越低越干净)
  生僻激活率: 给定"生僻字 + 它的专属搭档"样本, 单独激活生僻字能否带出搭档
  搭档名次  : 搭档在最终激活序列里的名次 (越小越好, -1 = 没进榜)
  思考内容  : 人眼看的实际输出
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import CharBrain, Params, build_inhibition, build_weights  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PUNCT = set("，。、；：？！“”‘’（）《》〈〉【】—…·～,.!?;:\"'()[]")
SEEDS = ["苹果手机", "苹果香蕉", "北京上海", "今天天气不错", "学生学习成绩", "医生给病人看病",
         "汽车销售下降", "人工智能技术", "股票市场价格", "足球比赛结果"]
VARIANTS = [
    ("off  不压", dict(prior_mode="off")),
    ("sat  全频段压", dict(prior_mode="sat", prior_k=8000.0, prior_pow=0.6)),
    ("tail 只压尾巴", dict(prior_mode="tail", prior_floor=3000.0, prior_pow=0.6)),
]


def load_counts(path):
    z = np.load(path, allow_pickle=False)
    V = len(z["chars"])
    C = np.zeros((V, V), dtype=np.int32)
    C[z["rows"], z["cols"]] = z["data"]
    return C, [str(c) for c in z["chars"]], z["freq"]


def rare_samples(C, chars, freq, n=20, lo=50, hi=800, min_count=20):
    """找"生僻字 -> 高共现搭档"样本: 字本身很冷门, 但和搭档的共现很实。"""
    c2i = {c: i for i, c in enumerate(chars)}
    idx = np.nonzero((freq >= lo) & (freq <= hi))[0]
    rng = random.Random(7)
    rng.shuffle(list(idx))
    out = []
    for i in idx:
        if chars[i] in PUNCT:
            continue
        row = C[i]
        if row.max() < min_count:
            continue
        j = int(np.argmax(row))
        out.append((chars[i], chars[j], int(row[j]), int(freq[i]), int(freq[j])))
        if len(out) >= n:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default=os.path.join(HERE, "models", "counts_char_brain.npz"))
    ap.add_argument("--target", type=int, default=50)
    ap.add_argument("--samples", type=int, default=20)
    args = ap.parse_args()

    C, chars, freq = load_counts(args.counts)
    c2i = {c: i for i, c in enumerate(chars)}
    print(f"共现矩阵: V={len(chars)}, 非零={int((C>0).sum()):,}, 总字量={int(freq.sum()):,}")
    samples = rare_samples(C, chars, freq, n=args.samples)
    print(f"生僻样本 {len(samples)} 个 (字频 50~800, 与搭档共现 ≥20), 例如: "
          + ", ".join(f"{a}({fa})→{b}(共现{c})" for a, b, c, fa, fb in samples[:6]))

    base = Params()
    base.target_active = args.target
    base.steps = 20

    for name, kw in VARIANTS:
        p = Params(**{**vars(base), **kw})
        W, Wn, st, factors = build_weights(C, p, char_freq=freq)
        b = CharBrain(chars, W, Wn, build_inhibition(C, freq, p), p, {}, factors=factors)
        b.freq = freq

        # 1) 噪声率: 常用种子激活出来的字有多"生僻"
        noise, means = [], []
        for s in SEEDS:
            tr = b.think(s, steps=20)
            act = [c for c, _ in tr.final if c not in PUNCT]
            ids = [c2i[c] for c in act if c in c2i]
            if not ids:
                continue
            noise.append(float(np.mean([freq[i] < 3000 for i in ids])))
            means.append(float(np.mean([freq[i] for i in ids])))
        # 2) 生僻字能否带出搭档
        hit, ranks = 0, []
        for a, bb, cnt, fa, fb in samples:
            if a not in c2i or bb not in c2i:
                continue
            tr = b.think(a, steps=20)
            seq = [c for c, _ in tr.final]
            if bb in seq:
                hit += 1
                ranks.append(seq.index(bb))
            else:
                ranks.append(-1)
        print(f"\n===== {name}")
        print(f"  噪声率(激活字中字频<3000的占比) = {np.mean(noise):.3f}   "
              f"激活字平均字频 = {np.mean(means):,.0f}")
        print(f"  生僻字带出搭档: {hit}/{len(samples)}  搭档平均名次 = "
              f"{np.mean([r for r in ranks if r >= 0]) if hit else float('nan'):.1f}")
        for s in SEEDS[:5]:
            tr = b.think(s, steps=20)
            content = [c for c, _ in tr.final if c not in PUNCT]
            print(f"    [{s}] " + "".join(content[:34]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
