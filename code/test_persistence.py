# -*- coding: utf-8 -*-
"""思考的两个基本问题:
  A. 会不会漂移?  (逐步统计"还在主题范围内的比例")
  B. 会不会有字被**持续激活**? (每个字的点亮步数、从头亮到尾的字、稳定核心集合、每步更替量)

用法: python test_persistence.py --seeds 医 --steps 60
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


def topic_scope(b: CharBrain, seeds: str, hop1: int = 24, hop2: int = 24) -> set:
    """话题范围: 种子的强后继 + 这些后继的强后继 (两跳)。"""
    scope = {c for c in seeds if c in b.char2id}
    frontier = list(scope)
    for hop in range(2):
        nxt = set()
        for ch in frontier:
            i = b.char2id.get(ch)
            if i is None:
                continue
            row = b.W[i]
            idx = np.argsort(-row)[: (hop1 if hop == 0 else hop2)]
            for j in idx:
                if row[j] > 0:
                    nxt.add(b.chars[j])
        nxt -= scope
        scope |= nxt
        frontier = list(nxt)[:60]
    return scope


def trace_run(b: CharBrain, seeds: str, steps: int, hold: bool, scope: set,
              scope1: set | None = None):
    p = b.p
    ext = b.seed_vector(seeds)
    seed_ids = np.nonzero(ext > 0)[0]
    a = ext.copy()
    d0 = ext @ b.Wn
    theta = float(np.clip(np.quantile(d0[d0 > 0], max(0.0, 1 - p.target_active / b.V)) * 0.6,
                          p.theta_min, p.theta_max)) if (d0 > 0).any() else p.theta_init
    k = min(p.target_active, b.V)
    rows, dwell, active_hist = [], {}, []
    for t in range(steps):
        drive = a @ b.Wn
        s = float(a.sum())
        if p.gain_control > 0:
            drive = drive / (1.0 + p.gain_control * s)
        if p.use_inhibition and len(b.inh_idx):
            drive = drive - p.inh_gain * b._inhibition(a)
        if p.seed_steps and t >= p.seed_steps:
            ext_term = np.zeros(b.V, dtype=np.float32)
        else:
            ext_term = ext * p.seed_gain
        pot = p.decay * a + drive + ext_term
        theta_star = float(np.partition(pot, b.V - k)[b.V - k]) - p.act_eps * p.out_soft
        theta = float(np.clip((1 - p.theta_lr) * theta + p.theta_lr * theta_star,
                              p.theta_min, p.theta_max))
        a = np.clip((pot - theta) / p.out_soft, 0.0, 1.0)
        if hold and not (p.seed_steps and t >= p.seed_steps):
            a[seed_ids] = np.maximum(a[seed_ids], ext[seed_ids])
        idx = np.nonzero(a > p.act_eps)[0]
        chars = [b.chars[i] for i in idx]
        active_hist.append(set(chars))
        for c in chars:
            dwell[c] = dwell.get(c, 0) + 1
        in_topic = sum(1 for c in chars if c in scope)
        in_near = sum(1 for c in chars if c in (scope1 or scope))
        prev = active_hist[-2] if len(active_hist) > 1 else set()
        rows.append({
            "t": t, "n": len(chars), "in_topic": in_topic,
            "ratio": in_topic / max(len(chars), 1),
            "near": in_near / max(len(chars), 1),
            "new": len(set(chars) - prev), "left": len(prev - set(chars)),
            "top": sorted(chars, key=lambda c: -a[b.char2id[c]])[:8],
        })
    warm = steps // 4
    tail_sets = active_hist[warm:]
    all_on = [c for c, n in dwell.items() if n >= steps - warm]
    core = set.intersection(*tail_sets) if tail_sets else set()
    return rows, dwell, all_on, sorted(core), active_hist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "models", "char_brain.npz"))
    ap.add_argument("--seeds", default="医")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "persistence.json"))
    args = ap.parse_args()

    base = CharBrain.load(args.model)
    base.p.steps = args.steps
    scope = topic_scope(base, args.seeds)
    scope1 = set()
    for ch in args.seeds:
        i = base.char2id.get(ch)
        if i is None:
            continue
        row = base.W[i]
        idx = np.argsort(-row)[:60]
        scope1 |= {base.chars[j] for j in idx if row[j] > 0}
    scope1 |= {c for c in args.seeds if c in base.char2id}
    print(f"种子「{args.seeds}」  话题范围(两跳) {len(scope)} 字, 近邻(一跳) {len(scope1)} 字  "
          f"思考 {args.steps} 步")
    print("  近邻示例: " + "".join(sorted(scope1 - set(args.seeds))[:36]) + " ...\n")

    variants = [
        ("① 现行参数 (种子持续供能)", dict(ext_hold=True)),
        ("② 种子只点 1 步, 之后完全不再输入", dict(ext_hold=False, seed_steps=1)),
        ("③ 种子点 3 步后停", dict(ext_hold=False, seed_steps=3)),
        ("④ 增益控制关掉", dict(ext_hold=True, gain_control=0.0)),
        ("⑤ 记忆短 decay=0.2", dict(ext_hold=True, decay=0.2)),
        ("⑥ 记忆长 decay=0.7", dict(ext_hold=True, decay=0.7)),
        ("⑦ 更聚焦 target=12", dict(ext_hold=True, target_active=12)),
        ("⑧ 更发散 target=60", dict(ext_hold=True, target_active=60)),
    ]
    report = {}
    for name, kw in variants:
        b = copy.deepcopy(base)
        hold = kw.pop("ext_hold", True)
        for k, v in kw.items():
            setattr(b.p, k, v)
        rows, dwell, all_on, core, hist = trace_run(b, args.seeds, args.steps, hold,
                                                    scope, scope1)
        tail = rows[len(rows) // 3:]
        ratio = float(np.mean([r["ratio"] for r in tail]))
        near = float(np.mean([r["near"] for r in tail]))
        turnover = float(np.mean([r["new"] + r["left"] for r in tail]))
        top_dwell = sorted(dwell.items(), key=lambda kv: -kv[1])[:12]
        report[name] = {
            "tail_ratio": ratio, "tail_near": near, "turnover": turnover,
            "dwell_top": top_dwell, "all_on": all_on, "core": core,
            "rows": rows,
        }
        print(f"== {name}")
        print(f"   后半程: 两跳范围内 {ratio:.2f} | 一跳近邻内 {near:.2f} "
              f"| 每步更替 {turnover:.0f} 字 | 最后一步激活 {rows[-1]['n']} 字")
        print(f"   从头亮到尾 {len(all_on)} 字: {''.join(all_on[:18]) or '（无）'}")
        print(f"   稳定核心 {len(core)} 字: {''.join(core[:20]) or '（无）'}")
        print(f"   点亮步数最多: " + ", ".join(f"{c}{n}" for c, n in top_dwell))
        print(f"   逐步(两跳比例): " + " ".join(f"{r['ratio']:.2f}" for r in rows))
        print(f"   最后一步内容: {''.join(rows[-1]['top'])}")
        print()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"seeds": args.seeds, "scope_size": len(scope), "scope": sorted(scope),
                   "variants": report}, f, ensure_ascii=False, indent=2, default=float)
    print(f"结果写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
