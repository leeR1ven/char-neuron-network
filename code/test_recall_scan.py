# -*- coding: utf-8 -*-
"""扫描: 同一句话"说几遍"、每遍学多快, 才能在只提一个概念时想起另一个?

对每个 (学习步长 η, 重复遍数 rounds) 组合:
  * 用"按语序逐字读入"的方式读这句话 (sequence=True), 每遍都实时学习;
  * 记录 连接强度 W(A→B) 与 传播权重 Ŵ(A→B);
  * 召回检验: 只激活 A, 看 B (及其邻居簇) 能不能被想起来、名次多少。
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import CharBrain  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PUNCT = set("，。、；：？！“”‘’（）《》〈〉【】—…·～,.!?;:\"'()[]")


def recall(b: CharBrain, seed: str, target: str, cluster: set, steps: int):
    tr = b.think(seed, steps=steps)
    seq = [c for c, _ in tr.final if c not in PUNCT]
    rank = seq.index(target) + 1 if target in seq else -1
    hits = [c for c in seq if c in cluster]
    return rank, hits, seq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "models", "char_brain.npz"))
    ap.add_argument("--sentence", default="医生用芯片给病人治病")
    ap.add_argument("--a", default="医")
    ap.add_argument("--b", default="芯")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--etas", default="0.004,0.01,0.03,0.06")
    ap.add_argument("--rounds", default="1,3,10,30")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "recall_scan.json"))
    args = ap.parse_args()

    base = CharBrain.load(args.model)
    base.p.steps = args.steps
    i, j = base.char2id[args.a], base.char2id[args.b]
    # B 的邻居簇 (学习前的语义邻居), 用于"想起了相关概念"的宽松判定
    row = base.W[j]
    idx = np.argsort(-row)[:12]
    cluster_b = {base.chars[x] for x in idx if row[x] > 0}
    row_a = base.W[i]
    idx_a = np.argsort(-row_a)[:12]
    cluster_a = {base.chars[x] for x in idx_a if row_a[x] > 0}
    print(f"句: 「{args.sentence}」  A={args.a} B={args.b} (steps={args.steps}, 逐字读入)")
    print(f"  A 的语义邻居: {''.join(sorted(cluster_a))}")
    print(f"  B 的语义邻居: {''.join(sorted(cluster_b))}")
    print(f"  学习前 W(A→B)={float(base.W[i,j]):.4f} Ŵ(A→B)={float(base.Wn[i,j]):.6f}")

    etas = [float(x) for x in args.etas.split(",")]
    rounds_list = [int(x) for x in args.rounds.split(",")]
    rows = []
    print(f"\n{'η':>7} {'遍数':>4} {'W(A→B)':>8} {'Ŵ(A→B)':>9} {'B名次':>6} "
          f"{'想起的B簇字':<14} 思考内容")
    for eta in etas:
        for rounds in rounds_list:
            b = copy.deepcopy(base)
            b.p.learn_eta = eta
            for _ in range(rounds):
                b.think(args.sentence, learn=True, steps=args.steps, sequence=True)
            w = float(b.W[i, j])
            wn = float(b.Wn[i, j])
            rank, hits, seq = recall(b, args.a, args.b, cluster_b, args.steps)
            rows.append({"eta": eta, "rounds": rounds, "W": w, "Wn": wn,
                         "rank_B": rank, "hits_B": hits, "content": seq[:30]})
            print(f"{eta:7.3f} {rounds:4d} {w:8.4f} {wn:9.6f} {rank:6d} "
                  f"{''.join(hits[:12]) or '（无）':<14} {''.join(seq[:22])}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"sentence": args.sentence, "A": args.a, "B": args.b, "rows": rows},
                  f, ensure_ascii=False, indent=2, default=float)
    print(f"\n结果写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
