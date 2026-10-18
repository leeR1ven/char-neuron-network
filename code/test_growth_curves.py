# -*- coding: utf-8 -*-
"""诊断 + 对比: 权重增长曲线的分布形式对"想到什么"的影响。

要回答的问题:
  1. 通用字(的/和/在/是/…)的权重到底有多高? 占比多少?
  2. 两个概念一起激活时, 出来的是"与两者都相关的中间概念字", 还是通用字?
  3. 换一种增长分布 (先快涨后回落, 类似正态/对数正态) 会不会改善?

指标:
  generic  激活集合里通用字(语料最高频前 200)的占比  ← 越低越好
  bridge   中间概念字: 与两个概念都有关联、但不属于任何一簇的字 ← 越多越好
  junk     生僻杂字(语料最冷门后 2000)占比
  recall   只说A时, B概念露头的字数
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import CharBrain, Params, build_inhibition, build_weights  # noqa: E402
from rebuild import load_counts  # noqa: E402
from test_concept_recall2 import (CASES, activation_vector, background,  # noqa: E402
                                  concept_chars)

HERE = os.path.dirname(os.path.abspath(__file__))
PUNCT = set("，。、；：？！“”‘’（）《》〈〉【】—…·～,.!?;:\"'()[]")


def curve_table(p: Params, counts=(1, 3, 10, 30, 100, 1000, 10000, 100000, 1000000,
                                   10000000)):
    from brain import growth_curve
    c = np.array(counts, dtype=np.float32)
    return dict(zip(counts, [round(float(x), 4) for x in growth_curve(c, p)]))


def evaluate(b: CharBrain, C: np.ndarray, freq: np.ndarray, steps: int, bg: np.ndarray,
             generic_ids: set, junk_ids: set):
    """两个概念一起激活 + 只说一个概念, 各测一次。"""
    out = {}
    for tag, sent, core_a, core_b in CASES:
        cA = concept_chars(b, core_a)
        cB = concept_chars(b, core_b)
        iA = [b.char2id[c] for c in cA if c in b.char2id]
        iB = [b.char2id[c] for c in cB if c in b.char2id]
        # 中间概念字: 与 A、B 两簇都有实共现的外人
        outside = [i for i in range(b.V) if i not in set(iA) | set(iB)
                   and b.chars[i] not in PUNCT]
        bridge = [i for i in outside
                  if C[i, iA].max() >= 15 and C[i, iB].max() >= 15]
        # --- 两个概念一起激活 ---
        both = "".join(core_a) + "".join(core_b)
        a_both = activation_vector(b, both, steps)
        order = np.argsort(-(a_both - bg))
        names = [b.chars[i] for i in order if b.chars[i] not in PUNCT][:40]
        gen = sum(1 for c in names if b.char2id[c] in generic_ids) / len(names)
        jnk = sum(1 for c in names if b.char2id[c] in junk_ids) / len(names)
        br = [b.chars[i] for i in bridge if b.char2id[b.chars[i]] in
              {b.char2id[c] for c in names}]
        # --- 只说 A ---
        aA = activation_vector(b, "".join(core_a), steps)
        dA = aA - bg
        namesA = [b.chars[i] for i in np.argsort(-dA) if b.chars[i] not in PUNCT][:30]
        hits = [c for c in namesA if c in cB]
        out[tag] = {"generic": gen, "junk": jnk, "bridge_hits": len(br),
                    "bridge_chars": br[:12], "recall_hits": len(hits),
                    "recall_chars": hits, "both_content": names[:34],
                    "cue_content": namesA[:28]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default=os.path.join(HERE, "models", "counts_char_brain.npz"))
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "growth_curves.json"))
    args = ap.parse_args()

    C, chars, freq = load_counts(args.counts)
    order = np.argsort(-freq)
    generic_ids = set(order[:200].tolist())
    junk_ids = set(order[-2000:].tolist())
    print(f"共现矩阵 V={len(chars)}  通用字(前200): "
          f"{''.join(chars[i] for i in order[:30])} ...")
    print(f"通用字平均出边强度占比检查: ", end="")

    VARIANTS = [
        ("① log + 归一化 (以前的做法)", dict(growth="log", k_sat=1.0, c_ref=10000.0,
                                            use_norm=True, norm_alpha=0.75)),
        ("② hill 不设拐点 (cut=1e9)", dict(growth="hill", rise_k=30.0, rise_a=0.5,
                                           cut_c=1e9, cut_b=1.0, use_norm=False)),
        ("③ hill 拐点 3e6 (很轻)", dict(growth="hill", rise_k=30.0, rise_a=0.5,
                                        cut_c=3e6, cut_b=1.0, use_norm=False)),
        ("④ hill 拐点 1e6 (轻)", dict(growth="hill", rise_k=30.0, rise_a=0.5,
                                      cut_c=1e6, cut_b=1.0, use_norm=False)),
        ("⑤ hill 拐点 1e5 (重)", dict(growth="hill", rise_k=30.0, rise_a=0.5,
                                      cut_c=1e5, cut_b=1.0, use_norm=False)),
        ("⑥ hill 拐点1e6 + 出边64", dict(growth="hill", rise_k=30.0, rise_a=0.5,
                                         cut_c=1e6, cut_b=1.0, use_norm=False, top_k=64)),
    ]
    report = {}
    base_brain = None
    for name, kw in VARIANTS:
        p = Params(**kw)
        p.target_active = 25
        p.steps = args.steps
        p.learn_gap_decay = 1.0
        W, Wn, st, factors = build_weights(C, p, char_freq=freq)
        b = CharBrain(chars, W, Wn, build_inhibition(C, freq, p), p, {}, factors=factors)
        b.freq = freq
        bg = background(b, args.steps)
        res = evaluate(b, C, freq, args.steps, bg, generic_ids, junk_ids)
        g = float(np.mean([r["generic"] for r in res.values()]))
        j = float(np.mean([r["junk"] for r in res.values()]))
        br = float(np.mean([r["bridge_hits"] for r in res.values()]))
        rc = float(np.mean([r["recall_hits"] for r in res.values()]))
        # 通用字的出边权重有多高
        gw = float(np.mean([W[i][W[i] > 0].mean() for i in list(generic_ids)]))
        allw = float(W[W > 0].mean())
        print(f"\n===== {name}  (峰值≈{max(curve_table(p).values()):.2f})")
        print(f"  曲线: {curve_table(p)}")
        print(f"  通用字出边平均权重 {gw:.3f} vs 全网络平均 {allw:.3f} (比值 {gw/allw:.2f})")
        nz = W[W > 0]
        key = []
        for x, y in (("手", "机"), ("机", "器"), ("医", "院"), ("苹", "果"), ("的", "是")):
            ii, jj = chars.index(x), chars.index(y)
            key.append(f"{x}{y}:{W[ii, jj]:.2f}" if W[ii, jj] > 0 else f"{x}{y}:x")
        print(f"  权重分布 p50={np.percentile(nz,50):.3f} p99={np.percentile(nz,99):.3f} "
              f"max={nz.max():.3f} | 关键搭配 " + " ".join(key))
        print(f"  通用字占比 {g:.2f} | 生僻杂字占比 {j:.2f} | 中间概念字 {br:.1f} 个 "
              f"| 只说A带出B {rc:.1f} 字")
        for tag, r in list(res.items())[:3]:
            print(f"    [{tag}] 一起激活: {''.join(r['both_content'][:30])}")
            print(f"        中间概念字: {''.join(r['bridge_chars'])}")
        report[name] = {"curve": curve_table(p), "generic_ratio": g, "junk_ratio": j,
                        "bridge_mean": br, "recall_mean": rc,
                        "generic_edge_w": gw, "all_edge_w": allw, "detail": res}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n结果写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
