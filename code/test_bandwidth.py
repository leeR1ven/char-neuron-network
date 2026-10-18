# -*- coding: utf-8 -*-
"""验证: 绑定生效了, 只是被"激活规模"卡住 —— 放宽带宽, 另一个概念就露头。

对每个案例: 绑定学习后, 只说概念A, 分别在 target_active = 15/25/40/60/100 下跑,
看概念B有多少字进入激活集合 (对照: 学习前的同一设置)。
"""
from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import CharBrain  # noqa: E402
from test_concept_recall2 import CASES, activation_vector, background, concept_chars  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = [15, 25, 40, 60, 100]


def hits_at(b: CharBrain, cue: str, other: set, bg: np.ndarray, steps: int, target: int):
    b.p.target_active = target
    a = activation_vector(b, cue, steps)
    d = a - bg
    order = np.argsort(-d)
    names = [b.chars[i] for i in order if b.chars[i] not in "，。、；：？！“”‘’（）《》〈〉【】—…·～,.!?;:\"'()[]"][:target]
    hits = [c for c in names if c in other]
    ids = [b.char2id[c] for c in other if c in b.char2id]
    mass = float(np.maximum(d[ids], 0).sum()) if ids else 0.0
    return hits, mass, names


def main() -> int:
    base = CharBrain.load(sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, 'models', 'char_brain.npz'))
    steps, eta, rounds = 24, float(sys.argv[1]) if len(sys.argv) > 1 else 0.05, \
        int(sys.argv[2]) if len(sys.argv) > 2 else 5
    base.p.steps = steps
    bg = background(base, steps)
    out = []
    for tag, sent, core_a, core_b in CASES:
        cA = concept_chars(base, core_a)
        cB = concept_chars(base, core_b)
        cue_a = "".join(core_a)
        print("=" * 78)
        print(f"【{tag}】只说A「{cue_a}」→ 概念B({len(cB)}字) 露出多少")
        b = copy.deepcopy(base)
        b.p.learn_eta = eta
        b.p.learn_gap_decay = 1.0
        for _ in range(rounds):
            b.think(sent, learn=True, steps=max(steps, len(sent) + 6), sequence=True)
        rows = []
        for tgt in TARGETS:
            base.p.target_active = tgt
            h0, m0, _ = hits_at(base, cue_a, cB, bg, steps, tgt)
            h1, m1, n1 = hits_at(b, cue_a, cB, bg, steps, tgt)
            rows.append({"target": tgt, "before": h0, "after": h1, "mass_after": m1})
            print(f"  target={tgt:3d}: 学习前带出B {len(h0):2d} 字 {''.join(h0)[:16]:16s} | "
                  f"学习后 {len(h1):2d} 字 {''.join(h1)[:16]:16s} | B质量 {m1:6.2f}")
        # 学习后末态内容 (target=60)
        _, _, names = hits_at(b, cue_a, cB, bg, steps, 60)
        print(f"  学习后 target=60 的思考: {''.join(names[:50])}")
        out.append({"tag": tag, "rows": rows})
    with open(os.path.join(HERE, "results", "bandwidth_test.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
