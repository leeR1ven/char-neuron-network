# -*- coding: utf-8 -*-
"""共用时长 (decay) 对"概念绑定"的影响 —— 这是参数, 不是新机制。

一个字亮着的窗口越长, 两个概念的字"同时在场"的机会越多,
时序赫布自然就能把两个方向的连接都学到。
"""
from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import CharBrain  # noqa: E402
from diag_concept_drive import drive_stats  # noqa: E402
from test_concept_recall2 import CASES, activation_vector, background, concept_chars  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PUNCT = set("，。、；：？！“”‘’（）《》〈〉【】—…·～,.!?;:\"'()[]")


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "models", "char_brain.npz")
    decays = [float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0.45,0.7,0.85").split(",")]
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    steps, eta = 26, 0.05
    base = CharBrain.load(model)
    base.p.steps = steps
    bg = background(base, steps)
    out = []
    for decay in decays:
        print("=" * 80)
        print(f"### decay = {decay} (一个字能亮多久: 衰减到 1/e 约需 "
              f"{1/max(1-decay,1e-6):.1f} 步)")
        tot_f = tot_b = 0
        for tag, sent, core_a, core_b in CASES:
            pa, pb = sent.find(core_a[0]), sent.find(core_b[0])
            first, second = (core_a, core_b) if pa < pb else (core_b, core_a)
            cF, cS = concept_chars(base, first), concept_chars(base, second)
            b = copy.deepcopy(base)
            b.p.learn_eta = eta
            b.p.decay = decay
            b.p.steps = max(steps, len(sent) + 6)
            for _ in range(rounds):
                b.think(sent, learn=True, steps=b.p.steps, sequence=True)
            b.p.steps = steps
            for name, cue, other, label in ((first, "".join(first), cS, "正向"),
                                            (second, "".join(second), cF, "反向")):
                a = activation_vector(b, cue, steps) - bg
                names = [b.chars[i] for i in np.argsort(-a) if b.chars[i] not in PUNCT][:30]
                hits = [c for c in names if c in other]
                if label == "正向":
                    tot_f += len(hits)
                else:
                    tot_b += len(hits)
                print(f"  [{tag}] {label}: 带出 {len(hits):2d} 字 {''.join(hits)}")
                out.append({"decay": decay, "tag": tag, "dir": label, "hits": hits,
                            "content": names[:26]})
        print(f"  → decay={decay} 合计: 正向 {tot_f} 字, 反向 {tot_b} 字")
    with open(os.path.join(HERE, "results", "decay_binding.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
