# -*- coding: utf-8 -*-
"""定量诊断: 绑定学习之后, 只提概念A时, 概念B到底收到多少驱动? 离阈值差多少?

对每个案例:
  1. 定义概念A/B (代表字 + 强关联字);
  2. 只说A跑一轮, 记录: A/B 两簇各自收到的总驱动、最高驱动、当前阈值 θ;
  3. 用绑定句学习若干遍, 再跑一次, 看 B 簇的驱动增加了多少、还差多少;
  4. 结论: 差在"驱动总量"还是"稀释"。
"""
from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import CharBrain  # noqa: E402
from test_concept_recall2 import CASES, activation_vector, concept_chars  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def drive_stats(b: CharBrain, cue: str, cluster: set, steps: int):
    """跑一轮, 返回簇内的驱动统计 (用最后一步的驱动)。"""
    p = b.p
    ext = b.seed_vector(cue)
    seed_ids = np.nonzero(ext > 0)[0]
    a = ext.copy()
    d0 = ext @ b.Wn
    theta = float(np.clip(np.quantile(d0[d0 > 0], max(0.0, 1 - p.target_active / b.V)) * 0.6,
                          p.theta_min, p.theta_max)) if (d0 > 0).any() else p.theta_init
    k = min(p.target_active, b.V)
    drive = np.zeros(b.V, dtype=np.float32)
    for _ in range(steps):
        drive = a @ b.Wn
        s = float(a.sum())
        if p.gain_control > 0:
            drive = drive / (1.0 + p.gain_control * s)
        if p.use_inhibition and len(b.inh_idx):
            drive = drive - p.inh_gain * b._inhibition(a)
        pot = p.decay * a + drive + ext * p.seed_gain
        ts = float(np.partition(pot, b.V - k)[b.V - k]) - p.act_eps * p.out_soft
        theta = float(np.clip((1 - p.theta_lr) * theta + p.theta_lr * ts, p.theta_min, p.theta_max))
        a = np.clip((pot - theta) / p.out_soft, 0.0, 1.0)
        a[seed_ids] = np.maximum(a[seed_ids], ext[seed_ids])
    ids = [b.char2id[c] for c in cluster if c in b.char2id]
    d = drive[ids] if ids else np.zeros(1)
    return {"sum": float(d.sum()), "max": float(d.max()), "theta": theta,
            "top": sorted(((b.chars[i], float(drive[i])) for i in ids), key=lambda x: -x[1])[:8]}


def main() -> int:
    base = CharBrain.load(os.path.join(HERE, "models", "char_brain.npz"))
    steps = 24
    base.p.steps = steps
    eta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    out = []
    for tag, sent, core_a, core_b in CASES:
        cA = concept_chars(base, core_a)
        cB = concept_chars(base, core_b)
        cue_a = "".join(core_a)
        before = drive_stats(base, cue_a, cB, steps)
        b = copy.deepcopy(base)
        b.p.learn_eta = eta
        b.p.learn_gap_decay = 1.0
        for _ in range(rounds):
            b.think(sent, learn=True, steps=max(steps, len(sent) + 6), sequence=True)
        after = drive_stats(b, cue_a, cB, steps)
        base_a = drive_stats(base, cue_a, cA, steps)
        after_a = drive_stats(b, cue_a, cA, steps)
        print("=" * 78)
        print(f"【{tag}】只说A「{cue_a}」  阈值 θ≈{after['theta']:.4f}")
        print(f"  概念B收到的驱动: 学习前 合计{before['sum']:.3f}/最高{before['max']:.4f}  →  "
              f"学习后 合计{after['sum']:.3f}/最高{after['max']:.4f}")
        print(f"  概念A收到的驱动: 学习前 合计{base_a['sum']:.3f}  →  学习后 合计{after_a['sum']:.3f}")
        gap = after["max"] - after["theta"]
        print(f"  B 最高的那个字离阈值: {gap:+.4f} "
              f"({'已能越过' if gap > 0 else '仍差 %.1f 倍' % (after['theta'] / max(after['max'], 1e-9))})")
        print(f"  B 簇里驱动最高的几个: " + ", ".join(f"{c}{v:.4f}" for c, v in after["top"]))
        out.append({"tag": tag, "theta": after["theta"], "B_before": before, "B_after": after,
                    "A_before": base_a, "A_after": after_a})
    with open(os.path.join(HERE, "results", "concept_drive_diag.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
