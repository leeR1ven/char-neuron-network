# -*- coding: utf-8 -*-
"""思考实验: 激活几个字, 看网络自己扩散出哪些字。

用法示例:
    python think.py --model models/char_brain.npz --seeds 苹果手机
    python think.py --model models/char_brain.npz --seeds 苹果香蕉 --no-inh
    python think.py --model models/char_brain.npz --seeds 今天天气不错我想出去走走 --learn
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import CharBrain, Params  # noqa: E402


def load_model(path: str) -> CharBrain:
    b = CharBrain.load(path)
    if not hasattr(b, "freq"):
        b.freq = None
    return b


PUNCT = set("，。、；：？！“”‘’（）《》〈〉【】—…·～,.!?;:\"'()[]")


def show_trace(brain: CharBrain, tr, topn: int = 22, title: str = "") -> None:
    print("=" * 78)
    print(title or ("种子: " + " ".join(tr.seeds)))
    for s in tr.steps:
        top = " ".join(c for c, _ in s.top[:topn])
        print(f"  t={s.t:02d} θ={s.theta:.4f} 激活{s.n_active:4d}字 | {top}")
    print("-" * 78)
    raw = " ".join(f"{c}{v:.2f}" for c, v in tr.final[:40])
    print("最终激活 (按强度): " + raw)
    content = [c for c, _ in tr.final if c not in PUNCT]
    print(f"思考内容 (去掉标点, {len(content)}字): " + "".join(content[:60]))
    order = [c for c, _, v in tr.order if c not in PUNCT and v > 0.3]
    print(f"按激活先后顺序: " + "".join(order[:60]))
    if tr.learned_edges:
        print("本次新增/增强的连接:")
        for i, j, before, after in sorted(tr.learned_edges, key=lambda x: -x[3])[:10]:
            print(f"    {i}->{j}: {before:.3f} -> {after:.3f}")
    print()


def overlap(a, b, k: int = 30):
    sa = {c for c, _ in a[:k]}
    sb = {c for c, _ in b[:k]}
    return sa & sb, sa - sb, sb - sa


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "models", "char_brain.npz"))
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--target-active", type=int, default=None)
    ap.add_argument("--theta-lr", type=float, default=None)
    ap.add_argument("--inh-gain", type=float, default=None)
    ap.add_argument("--no-inh", action="store_true")
    ap.add_argument("--learn", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--top", type=int, default=22)
    args = ap.parse_args()

    brain = load_model(args.model)
    p = brain.p
    if args.steps:
        p.steps = args.steps
    if args.target_active:
        p.target_active = args.target_active
    if args.theta_lr is not None:
        p.theta_lr = args.theta_lr
    if args.inh_gain is not None:
        p.inh_gain = args.inh_gain
    if args.no_inh:
        p.use_inhibition = False

    seeds = list(args.seeds)
    tr = brain.think(seeds, learn=args.learn)
    print(f"模型: {args.model}  神经元 {brain.V} 个  "
          f"(抑制={'开' if p.use_inhibition else '关'}, 目标激活数={p.target_active}, "
          f"阈值步长={p.theta_lr})")
    show_trace(brain, tr, args.top)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(tr.to_json(), f, ensure_ascii=False, indent=2)
        print(f"轨迹已写入 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
