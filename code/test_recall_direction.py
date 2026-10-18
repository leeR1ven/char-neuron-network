# -*- coding: utf-8 -*-
"""按"学习方向"检验召回: 绑定句里谁先出现, 就应该由谁把对方想起来。

同时给出定量: 只提先出现的概念时, 对方概念收到的最高驱动 vs 阈值。
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
    model = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "models", "char_brain_hill.npz")
    eta = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    steps = 24
    base = CharBrain.load(model)
    base.p.steps = steps
    bg = background(base, steps)
    rows = []
    for tag, sent, core_a, core_b in CASES:
        # 判断谁在句子里先出现
        pa, pb = sent.find(core_a[0]), sent.find(core_b[0])
        first, second = (core_a, core_b) if pa < pb else (core_b, core_a)
        fname, sname = ("A", "B") if pa < pb else ("B", "A")
        cF = concept_chars(base, first)
        cS = concept_chars(base, second)
        print("=" * 80)
        print(f"【{tag}】句中顺序: 「{first[0]}」在前(第{min(pa,pb)}字), "
              f"「{second[0]}」在后(第{max(pa,pb)}字) → 学习方向 {fname}→{sname}")

        b = copy.deepcopy(base)
        b.p.learn_eta = eta
        b.p.learn_gap_decay = 1.0
        for _ in range(rounds):
            b.think(sent, learn=True, steps=max(steps, len(sent) + 6), sequence=True)

        cue_f, cue_s = "".join(first), "".join(second)
        # 方向 1: 先出现的概念 -> 想后出现的概念
        d0 = drive_stats(base, cue_f, cS, steps)
        d1 = drive_stats(b, cue_f, cS, steps)
        a1 = activation_vector(b, cue_f, steps) - bg
        names = [b.chars[i] for i in np.argsort(-a1) if b.chars[i] not in PUNCT][:30]
        hits = [c for c in names if c in cS]
        # 方向 2: 反着来 (不应该更容易)
        d2 = drive_stats(b, cue_s, cF, steps)
        a2 = activation_vector(b, cue_s, steps) - bg
        names2 = [b.chars[i] for i in np.argsort(-a2) if b.chars[i] not in PUNCT][:30]
        hits2 = [c for c in names2 if c in cF]
        print(f"  ✅ 正向检验: 只说「{cue_f}」→ 后出现的概念({len(cS)}字)")
        print(f"      对方最高驱动 {d0['max']:.4f} → {d1['max']:.4f} (阈值 θ≈{d1['theta']:.4f}); "
              f"簇总驱动 {d0['sum']:.2f} → {d1['sum']:.2f}")
        print(f"      带出对方 {len(hits)} 字: {''.join(hits)}")
        print(f"      思考: {''.join(names[:30])}")
        print(f"  ↩︎ 反向检验: 只说「{cue_s}」→ 先出现的概念({len(cF)}字): "
              f"带出 {len(hits2)} 字 {''.join(hits2)} | 最高驱动 {d2['max']:.4f}")
        rows.append({"tag": tag, "first": fname, "second": sname, "sent": sent,
                     "fwd_max": [d0["max"], d1["max"]], "fwd_sum": [d0["sum"], d1["sum"]],
                     "theta": d1["theta"], "fwd_hits": hits, "bwd_hits": hits2,
                     "fwd_content": names[:30], "bwd_content": names2[:30]})
    with open(os.path.join(HERE, "results", "recall_by_direction.json"), "w",
              encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n结果写入 results/recall_by_direction.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
