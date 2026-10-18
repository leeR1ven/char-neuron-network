# -*- coding: utf-8 -*-
"""新思想 + 实时学习 + 整句召回 的正式实验。

流程:
  阶段1 学习: 说一句"两个本来很远的概念"合在一起的话 (按语序逐字读入, 实时学习, 重复若干遍);
  阶段2 检验: 换一整句话当线索 —— 只说概念A那边的日常话 (不含B的任何字),
             看网络会不会自己把概念B那边的字带出来;
             反向同理 (只说B那边的话, 看能否带出A那边的字);
  对照: 学习之前的同一个线索句, 能不能带出对方 (作为基线)。

判定"想起来"的口径放宽到**概念簇**: 对方那个字本身, 或它最亲近的邻居字, 出现即算。
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

# (标签, 学习句, 概念A字, 概念B字, 线索句A(只谈A), 线索句B(只谈B))
CASES = [
    ("医学×芯片",
     "医生用芯片给病人治病", "医", "芯",
     "医院里的医生正在给病人治疗", "芯片的电路板和材料"),
    ("野兽×金融",
     "老虎在银行里存了很多钱", "虎", "银",
     "动物园里的老虎和狮子", "银行存款和利息"),
    ("天体×水果",
     "月亮的表面种出了苹果", "月", "苹",
     "月球表面和星星", "苹果树和水果"),
    ("机器×文学",
     "机器人也开始读诗写文章", "机", "诗",
     "机器和自动化生产", "诗歌和文学作品"),
]


def cluster_of(b: CharBrain, ch: str, k: int = 12) -> set:
    i = b.char2id.get(ch)
    if i is None:
        return set()
    row = b.W[i]
    idx = np.argsort(-row)[:k]
    s = {b.chars[j] for j in idx if row[j] > 0}
    s.add(ch)
    return s


def run_cue(b: CharBrain, cue: str, target_cluster: set, target_ch: str, steps: int):
    tr = b.think(cue, steps=steps)
    seq = [c for c, _ in tr.final if c not in PUNCT]
    hits = [c for c in seq if c in target_cluster]
    rank = seq.index(target_ch) + 1 if target_ch in seq else -1
    return {"content": seq[:34], "hits": hits, "rank": rank,
            "n_mean": tr.n_mean, "theta": tr.theta_end}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "models", "char_brain.npz"))
    ap.add_argument("--eta", type=float, default=0.03)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "newidea_formal.json"))
    args = ap.parse_args()

    base = CharBrain.load(args.model)
    base.p.steps = args.steps
    print(f"模型 V={base.V}  学习步长 η={args.eta}  重复 {args.rounds} 遍  "
          f"每遍 {args.steps} 步 (按语序逐字读入)\n")

    report = []
    for tag, sent, a_ch, b_ch, cue_a, cue_b in CASES:
        print("=" * 84)
        print(f"【{tag}】学习句: 「{sent}」")
        cA, cB = cluster_of(base, a_ch), cluster_of(base, b_ch)
        print(f"  概念A「{a_ch}」语义邻居(学习前): {''.join(sorted(cA - {a_ch}))}")
        print(f"  概念B「{b_ch}」语义邻居(学习前): {''.join(sorted(cB - {b_ch}))}")

        # 基线: 学习之前, 线索句能不能带出对方
        base_a = run_cue(base, cue_a, cB, b_ch, args.steps)
        base_b = run_cue(base, cue_b, cA, a_ch, args.steps)
        print(f"  [基线] 线索句A「{cue_a}」→ 带出B簇 {len(base_a['hits'])} 字 "
              f"{''.join(base_a['hits'][:10]) or '（无）'}")
        print(f"  [基线] 线索句B「{cue_b}」→ 带出A簇 {len(base_b['hits'])} 字 "
              f"{''.join(base_b['hits'][:10]) or '（无）'}")

        # 学习
        b = copy.deepcopy(base)
        b.p.learn_eta = args.eta
        i, j = b.char2id[a_ch], b.char2id[b_ch]
        w0 = (float(b.W[i, j]), float(b.W[j, i]))
        for _ in range(args.rounds):
            b.think(sent, learn=True, steps=args.steps, sequence=True)
        w1 = (float(b.W[i, j]), float(b.W[j, i]))
        # 两个概念簇之间新长出的连接
        cross = []
        for x in sorted(cA):
            xi = b.char2id.get(x)
            if xi is None:
                continue
            for y in sorted(cB):
                yi = b.char2id.get(y)
                if yi is None:
                    continue
                if b.W[xi, yi] > 0.05:
                    cross.append((x, y, float(b.W[xi, yi])))
        cross.sort(key=lambda e: -e[2])
        print(f"  学习后: W({a_ch}→{b_ch})={w0[0]:.3f}→{w1[0]:.3f}, "
              f"W({b_ch}→{a_ch})={w0[1]:.3f}→{w1[1]:.3f}; "
              f"两簇之间形成 {len(cross)} 条连接")
        print(f"    最强的跨簇连接: " + ", ".join(f"{x}→{y}({w:.2f})" for x, y, w in cross[:8]))

        # 检验: 整句线索
        af = run_cue(b, cue_a, cB, b_ch, args.steps)
        bf = run_cue(b, cue_b, cA, a_ch, args.steps)
        print(f"  [学习后] 线索句A「{cue_a}」→ 带出B簇 {len(af['hits'])} 字 "
              f"{''.join(af['hits'][:10]) or '（无）'}"
              f"{'  ★想起了 ' + b_ch if af['rank'] > 0 else ''}")
        print(f"      思考内容: {''.join(af['content'][:30])}")
        print(f"  [学习后] 线索句B「{cue_b}」→ 带出A簇 {len(bf['hits'])} 字 "
              f"{''.join(bf['hits'][:10]) or '（无）'}"
              f"{'  ★想起了 ' + a_ch if bf['rank'] > 0 else ''}")
        print(f"      思考内容: {''.join(bf['content'][:30])}")

        report.append({
            "tag": tag, "sentence": sent, "A": a_ch, "B": b_ch,
            "cue_A": cue_a, "cue_B": cue_b,
            "cluster_A": sorted(cA), "cluster_B": sorted(cB),
            "w_before": w0, "w_after": w1, "cross_edges": cross[:30],
            "baseline_A_hits": base_a["hits"], "baseline_B_hits": base_b["hits"],
            "after_A_hits": af["hits"], "after_B_hits": bf["hits"],
            "after_A_rankB": af["rank"], "after_B_rankA": bf["rank"],
            "after_A_content": af["content"], "after_B_content": bf["content"],
            "learn_sentence_content": [c for c, _ in
                                       b.think(sent, steps=args.steps).final
                                       if c not in PUNCT][:34],
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n结果写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
