# -*- coding: utf-8 -*-
"""权重参数扫描 (在缓存好的共现矩阵上跑, 不用重读语料)。

扫的是"训练期"参数: 增长曲线 k_sat/c_ref、PMI 闸门、锐化 γ、归一化 α。
评价方式: 把每个配置训出来的网络拿去"思考"几组种子, 看话题命中 / 虚词占比 / 区分度。

用法:
    python sweep_weights.py --counts models/counts_smoke.npz --quick
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import CharBrain, Params, build_inhibition, build_weights  # noqa: E402
from sweep_think import FUNC, TOPIC, evaluate  # noqa: E402


def load_counts(path):
    z = np.load(path, allow_pickle=False)
    V = len(z["chars"])
    C = np.zeros((V, V), dtype=np.int32)
    C[z["rows"], z["cols"]] = z["data"]
    chars = [str(c) for c in z["chars"]]
    freq = z["freq"]
    return C, chars, freq


def build(C, chars, freq, p: Params, inh: bool = True):
    W, Wn, st, factors = build_weights(C, p, char_freq=freq)
    inhd = build_inhibition(C, freq, p) if inh else None
    b = CharBrain(chars, W, Wn, inhd, p, {}, factors=factors)
    b.freq = freq
    return b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     "models", "counts_smoke.npz"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    C, chars, freq = load_counts(args.counts)
    print(f"载入共现矩阵: V={len(chars)}, 非零={int((C>0).sum()):,}, 总计={int(C.sum()):,}")

    if args.quick:
        grid = list(itertools.product([3.0], [3000.0], [0.0, 0.35, 0.6], [1.5, 3.0]))
    else:
        grid = list(itertools.product([1.0, 3.0, 10.0], [1000.0, 3000.0, 20000.0],
                                      [0.0, 0.25, 0.5], [1.0, 2.0, 3.0]))
    print(f"组合数: {len(grid)}")

    rows = []
    for k_sat, c_ref, alpha, sharp in grid:
        t0 = time.time()
        p = Params(k_sat=k_sat, c_ref=c_ref, norm_alpha=alpha, sharp_pow=sharp,
                   min_freq=8, top_k=64, c_min=3)
        p.target_active = 50
        p.out_soft = 0.15
        p.steps = 16
        b = build(C, chars, freq, p, inh=True)
        recs = {}
        for seeds in TOPIC:
            recs[seeds] = evaluate(b, seeds, steps=16)
        topic = float(np.nanmean([r["topic"] for r in recs.values()]))
        func = float(np.nanmean([r["func"] for r in recs.values()]))
        n = float(np.nanmean([r["n"] for r in recs.values()]))
        a = set(recs["苹果手机"]["final"])
        bb = set(recs["苹果香蕉"]["final"])
        sep = len(a & bb) / max(len(a | bb), 1)
        rows.append((topic, func, n, sep, (k_sat, c_ref, alpha, sharp), recs))
        print(f"  K={k_sat:<5} Cref={c_ref:<7} α={alpha:<5} γ={sharp:<4} "
              f"topic={topic:.3f} func={func:.3f} n={n:.0f} sep={sep:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    rows.sort(key=lambda r: (-r[0], r[1]))
    print(f"\n{'topic':>6} {'func':>6} {'n':>6} {'sep':>5}  (K, Cref, α, γ)")
    for topic, func, n, sep, cfg, _ in rows[: args.top]:
        print(f"{topic:6.3f} {func:6.3f} {n:6.0f} {sep:5.2f}  {cfg}")
    best = rows[0]
    print("\n最佳配置思考示例:")
    for seeds, r in best[5].items():
        print(f"  [{seeds}] topic={r['topic']:.2f} func={r['func']:.2f} θ={r['theta_end']:.4f}")
        print("     " + " ".join(r["final"][:34]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
