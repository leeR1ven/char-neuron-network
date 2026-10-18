# -*- coding: utf-8 -*-
"""新思想 + 实时学习 检验。

流程 (对应"说一句话 -> 产生新想法 -> 下次只提一个概念 -> 能否想起另一个"):
  1) 取两个原本很远的概念 (字频都不低, 但彼此几乎不共现), 记下它们各自的"邻居簇";
  2) 说一句话, 把两个概念一起放进去, 让网络思考并实时学习 (可多轮, 模拟反复一起出现);
  3) 看这轮思考里长出了哪些**全新连接** (W 从 ~0 变成多少), 以及多激活了哪些字;
  4) 关键的召回检验: 下一次**只说第一个概念**, 看第二个概念(或它的邻居)能不能被想起来,
     名次比学习前提升多少; 反向也测一次 (只说第二个概念能否想起第一个)。

用法:
    python test_newidea.py --model models/char_brain.npz --rounds 3
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

# 每项: (一句话, 概念A, 概念B, 说明)
CASES = [
    ("医生用芯片给病人治病", "医", "芯", "医学 × 芯片"),
    ("老虎在银行里存了很多钱", "虎", "银", "野兽 × 金融"),
    ("月亮上种出了一片苹果", "月", "苹", "天体 × 水果"),
    ("机器人也开始读诗写文章", "机", "诗", "机器 × 文学"),
]


def neighbors(b: CharBrain, ch: str, k: int = 10):
    i = b.char2id.get(ch)
    if i is None:
        return []
    row = b.W[i].copy()
    idx = np.argsort(-row)[:k]
    return [b.chars[j] for j in idx if row[j] > 0]


def rank_of(b: CharBrain, seed: str, target: str, steps: int):
    """单独激活 seed, 看 target 在最终激活序列里的名次 (-1 = 没出现)。"""
    tr = b.think(seed, steps=steps)
    seq = [c for c, _ in tr.final]
    return (seq.index(target) if target in seq else -1), tr


def activate(b: CharBrain, seeds, steps: int):
    tr = b.think(seeds, steps=steps)
    return [c for c, _ in tr.final if c not in PUNCT], tr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "models", "char_brain.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--rounds", type=int, default=3, help="同一句话反复说几轮 (模拟反复共现)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--eta", type=float, default=None, help="单轮学习步长, 默认用模型里的值")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    base = CharBrain.load(args.model)
    if args.eta is not None:
        base.p.learn_eta = args.eta
    base.p.steps = args.steps
    print(f"模型 V={base.V}  连接={int((base.W>0).sum()):,}  "
          f"learn_eta={base.p.learn_eta}  每轮步数={args.steps}")

    report = []
    for sent, a_ch, b_ch, tag in CASES:
        i, j = base.char2id.get(a_ch), base.char2id.get(b_ch)
        if i is None or j is None:
            print(f"跳过 {tag}: 字不在词表里")
            continue
        print("\n" + "=" * 78)
        print(f"【{tag}】一句话: 「{sent}」")
        b = copy.deepcopy(base)
        w_ab0, w_ba0 = float(b.W[i, j]), float(b.W[j, i])
        ca, cb = neighbors(b, a_ch, 8), neighbors(b, b_ch, 8)
        print(f"  学习前: {a_ch}→{b_ch}={w_ab0:.4f}, {b_ch}→{a_ch}={w_ba0:.4f}")
        print(f"  {a_ch} 的邻居: {''.join(ca)}")
        print(f"  {b_ch} 的邻居: {''.join(cb)}")

        # 学习前: 单独激活 A / B 时对方的名次
        r_b_before, _ = rank_of(b, a_ch, b_ch, args.steps)
        r_a_before, _ = rank_of(b, b_ch, a_ch, args.steps)
        before_content, _ = activate(b, sent, args.steps)

        # ---- 反复说这句话, 每轮实时学习 ----
        learned_union: dict[tuple[str, str], float] = {}
        for r in range(args.rounds):
            tr = b.think(sent, learn=True, steps=args.steps)
            for x, y, wb, wa in tr.learned_edges:
                key = (x, y)
                if key not in learned_union:
                    learned_union[key] = wb
        w_ab1, w_ba1 = float(b.W[i, j]), float(b.W[j, i])
        print(f"  学习后: {a_ch}→{b_ch}={w_ab1:.4f} (+{w_ab1-w_ab0:.4f}), "
              f"{b_ch}→{a_ch}={w_ba1:.4f} (+{w_ba1-w_ba0:.4f})")

        # 这轮新长出来的连接 (跨两个概念簇的优先展示)
        new_edges = []
        for (x, y), wb in learned_union.items():
            wa = float(b.W[b.char2id[x], b.char2id[y]])
            if wa - wb > 0.02:
                new_edges.append((x, y, wb, wa))
        new_edges.sort(key=lambda e: -(e[3] - e[2]))
        cross = [e for e in new_edges
                 if (e[0] in ca + [a_ch] and e[1] in cb + [b_ch])
                 or (e[0] in cb + [b_ch] and e[1] in ca + [a_ch])]
        print(f"  本轮共新增/加强 {len(new_edges)} 条连接, 其中横跨两个概念簇的有 {len(cross)} 条:")
        for x, y, wb, wa in cross[:8]:
            print(f"     {x}→{y}: {wb:.3f} → {wa:.3f}")

        # 学习后再想这句话, 内容变了吗
        after_content, tr_after = activate(b, sent, args.steps)
        new_chars = [c for c in after_content if c not in before_content]
        print(f"  同一句话的思考内容变化: 新增 {' '.join(new_chars[:20]) or '（无）'}")

        # ---- 关键: 只提一个概念, 能否想起另一个 ----
        r_b_after, tr_a = rank_of(b, a_ch, b_ch, args.steps)
        r_a_after, tr_b = rank_of(b, b_ch, a_ch, args.steps)
        content_a = [c for c, _ in tr_a.final if c not in PUNCT]
        content_b = [c for c, _ in tr_b.final if c not in PUNCT]
        hit_a = [c for c in content_a if c in cb or c == b_ch]
        hit_b = [c for c in content_b if c in ca or c == a_ch]
        print(f"  召回检验 1: 只说「{a_ch}」→ {b_ch} 名次 {r_b_before} → {r_b_after}; "
              f"带出的对方簇字: {''.join(hit_a) or '（无）'}")
        print(f"     思考内容: {''.join(content_a[:26])}")
        print(f"  召回检验 2: 只说「{b_ch}」→ {a_ch} 名次 {r_a_before} → {r_a_after}; "
              f"带出的对方簇字: {''.join(hit_b) or '（无）'}")
        print(f"     思考内容: {''.join(content_b[:26])}")

        report.append({
            "tag": tag, "sentence": sent, "A": a_ch, "B": b_ch,
            "w_before": [w_ab0, w_ba0], "w_after": [w_ab1, w_ba1],
            "learned_edges_total": len(new_edges), "cross_edges": cross[:20],
            "rank_B_before": r_b_before, "rank_B_after": r_b_after,
            "rank_A_before": r_a_before, "rank_A_after": r_a_after,
            "recall_content_A": content_a[:30], "recall_content_B": content_b[:30],
            "recall_hits_A": hit_a, "recall_hits_B": hit_b,
            "content_before": before_content[:30], "content_after": after_content[:30],
            "new_chars": new_chars[:20],
        })

    out = os.path.join(args.out, "newidea_test.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n结果写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
