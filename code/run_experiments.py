# -*- coding: utf-8 -*-
"""完整实验: 检验这套字神经元网络的思考行为。

跑七组实验并把结果写成 JSON + Markdown:
  1. 阈值整体调控: 激活规模是否稳定在目标区间, θ 怎么动
  2. 上下文对比: 苹果手机 / 苹果香蕉 / 苹果价格 / 苹果公司 的思考范围差异
  3. 随机激活几个字
  4. 整句话激活
  5. 抑制神经元 开/关 对比
  6. 少见连接保护: 冷门字能不能被激活
  7. 新思想: 很少一起出现的字一起激活后, 连接被加强并长出新路径

用法: python run_experiments.py --model models/char_brain.npz --out results
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import CharBrain  # noqa: E402

FUNC = set("的是不我了和有为最在也就都而及与被把对从到着过得地吗呢吧啊这那你他它"
           "一二三四五六七八九十个年月日时分上下前后中大小多少好会很更还没只可以要"
           "，。、；：？！“”‘’（）《》〈〉【】—…·～")
PUNCT = set("，。、；：？！“”‘’（）《》〈〉【】—…·～,.!?;:\"'()[]")


def pretty(chars, filt_punct=True):
    if filt_punct:
        chars = [c for c in chars if c not in PUNCT]
    return "".join(chars)


def run(brain: CharBrain, seeds, steps=24, **kw):
    p = brain.p
    for k, v in kw.items():
        setattr(p, k, v)
    return brain.think(seeds, steps=steps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "models", "char_brain.npz"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "results"))
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    brain = CharBrain.load(args.model)
    p = brain.p
    random.seed(20261011)
    R: dict = {"model": os.path.basename(args.model), "meta": brain.meta,
               "params": {k: v for k, v in vars(p).items() if not k.startswith("_")}}

    def save_trace(name, tr, extra=None):
        fn = os.path.join(args.out, f"{name}.json")
        d = tr.to_json()
        if extra:
            d.update(extra)
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2, default=float)
        return d

    # ---------------------------------------------------------- 实验1 阈值调控
    print("[1/7] 阈值整体调控 ...")
    tr = brain.think("苹果手机", steps=args.steps)
    e1 = {
        "seeds": "苹果手机",
        "target_active": p.target_active,
        "n_active_hist": tr.n_active_hist,
        "theta_hist": [round(s.theta, 5) for s in tr.steps],
        "n_mean": tr.n_mean, "n_swing": tr.n_swing,
        "in_band": bool(abs(tr.n_mean - p.target_active) <= 0.5 * p.target_active),
    }
    save_trace("01_threshold", tr, {"summary": e1})
    R["experiment1_threshold"] = e1
    print(f"    激活数轨迹 {tr.n_active_hist}")
    print(f"    阈值轨迹   {[round(s.theta,4) for s in tr.steps]}")

    # ---------------------------------------------------------- 实验2 上下文对比
    print("[2/7] 上下文对比 ...")
    contexts = ["苹果手机", "苹果香蕉", "苹果价格", "苹果公司", "苹果发布会", "香蕉价格"]
    finals = {}
    for ctx in contexts:
        t = brain.think(ctx, steps=args.steps)
        finals[ctx] = [c for c, _ in t.final[:40] if c not in PUNCT]
        save_trace("02_ctx_" + ctx, t)
        print(f"    [{ctx}] n={t.n_mean:.0f}  " + pretty([c for c, _ in t.final[:30]]))
    base_a = set(finals["苹果手机"])
    base_b = set(finals["苹果香蕉"])
    cmp = {
        "sets": finals,
        "jaccard_手机_vs_香蕉": len(base_a & base_b) / max(len(base_a | base_b), 1),
        "共同": sorted(base_a & base_b),
        "只在手机出现": sorted(base_a - base_b),
        "只在香蕉出现": sorted(base_b - base_a),
    }
    R["experiment2_context"] = cmp
    with open(os.path.join(args.out, "02_context_compare.json"), "w", encoding="utf-8") as f:
        json.dump(cmp, f, ensure_ascii=False, indent=2, default=float)
    print(f"    重合度(Jaccard)={cmp['jaccard_手机_vs_香蕉']:.3f}")

    # ---------------------------------------------------------- 实验3 随机激活
    print("[3/7] 随机激活几个字 ...")
    vocab = [c for c in brain.chars if c not in PUNCT and c not in FUNC]
    rand_sets = []
    for i in range(3):
        seeds = random.sample(vocab, 4)
        t = brain.think(seeds, steps=args.steps)
        rand_sets.append({"seeds": "".join(seeds), "n_mean": t.n_mean,
                          "final": [c for c, _ in t.final[:30] if c not in PUNCT]})
        save_trace(f"03_random_{i}", t)
        print(f"    [{' '.join(seeds)}] -> " + pretty([c for c, _ in t.final[:26]]))
    R["experiment3_random"] = rand_sets

    # ---------------------------------------------------------- 实验4 整句激活
    print("[4/7] 整句话激活 ...")
    sentences = ["今天天气不错我想出去走走", "人工智能正在改变我们的生活方式",
                 "这家公司的手机销量超过了预期"]
    sent_res = []
    for s in sentences:
        t = brain.think(s, steps=args.steps)
        sent_res.append({"seeds": s, "n_mean": t.n_mean, "theta_end": t.theta_end,
                         "final": [c for c, _ in t.final[:40] if c not in PUNCT],
                         "order": [c for c, _, _ in t.order[:40] if c not in PUNCT]})
        save_trace("04_sentence_" + s[:6], t)
        print(f"    [{s}] -> " + pretty([c for c, _ in t.final[:34]]))
    R["experiment4_sentence"] = sent_res

    # ---------------------------------------------------------- 实验5 抑制神经元
    print("[5/7] 抑制神经元 开/关 ...")
    inh_cmp = {}
    for label, use_inh in (("开", True), ("关", False)):
        p.use_inhibition = use_inh
        t = brain.think("苹果手机", steps=args.steps)
        inh_cmp[label] = {"n_mean": t.n_mean, "theta_end": t.theta_end,
                          "final": [c for c, _ in t.final[:30] if c not in PUNCT],
                          "func_ratio": sum(1 for c, _ in t.final[:30] if c in FUNC) / 30}
        print(f"    抑制={label}: n={t.n_mean:.0f} 虚词占比={inh_cmp[label]['func_ratio']:.2f} "
              + pretty([c for c, _ in t.final[:24]]))
    p.use_inhibition = True
    R["experiment5_inhibition"] = inh_cmp

    # ---------------------------------------------------------- 实验6 少见连接保护
    print("[6/7] 少见连接保护 ...")
    freq = getattr(brain, "freq", None)
    rare_pairs = []
    if freq is not None:
        order = np.argsort(freq)          # 从最冷门到最常见
        rare_chars = [brain.chars[i] for i in order[:800] if brain.chars[i] not in PUNCT]
        for c in rare_chars:
            i = brain.char2id[c]
            row = brain.W[i]
            js = np.nonzero(row)[0]
            if len(js) == 0:
                continue
            j = js[np.argmax(row[js])]
            if row[j] >= 0.12:
                rare_pairs.append({"rare": c, "freq": int(freq[i]),
                                   "partner": brain.chars[j],
                                   "w": float(row[j]), "partner_freq": int(freq[j])})
            if len(rare_pairs) >= 10:
                break
    print(f"    找到 {len(rare_pairs)} 个[冷门字->搭档]样本")
    hit = 0
    demos = []
    for rp in rare_pairs[:5]:
        t = brain.think(rp["rare"], steps=args.steps)
        activated = [c for c, _ in t.final[:40]]
        ok = rp["partner"] in activated
        hit += int(ok)
        demos.append({**rp, "partner_activated": ok,
                      "final": [c for c in activated if c not in PUNCT][:25]})
        print(f"    冷门字 {rp['rare']}(频次{rp['freq']}) -> 搭档 {rp['partner']} "
              f"{'激活成功' if ok else '未激活'} | " + pretty(activated[:20]))
    R["experiment6_rare"] = {"pairs": rare_pairs, "demos": demos,
                             "hit_rate": hit / max(len(demos), 1)}

    # ---------------------------------------------------------- 实验7 新思想
    print("[7/7] 新思想: 一起激活 -> 长出连接 ...")
    # 每对都是"平时几乎不搭界"的单字 (香蕉 vs 手机, 月亮 vs 芯片, 老虎 vs 银行)
    setups = [("蕉", "机"), ("月", "芯"), ("虎", "银")]
    novel = []
    for a_ch, b_ch in setups:
        i, j = brain.char2id.get(a_ch), brain.char2id.get(b_ch)
        if i is None or j is None:
            continue
        before_ij = float(brain.W[i, j])
        before_ji = float(brain.W[j, i])
        # 学习前的名次 (单独激活 a, b 排第几)
        t0 = brain.think([a_ch], steps=args.steps)
        rank_before = next((n for n, (c, _) in enumerate(t0.final) if c == b_ch), -1)
        p.learn_eta = 0.004
        p.learn_pow = 1.0
        t = brain.think([a_ch, b_ch], steps=args.steps, learn=True)
        after_ij = float(brain.W[i, j])
        after_ji = float(brain.W[j, i])
        # 再想一次, 看新连接是否让两个字更容易互相激活
        t2 = brain.think([a_ch], steps=args.steps)
        rank_after = next((n for n, (c, _) in enumerate(t2.final) if c == b_ch), -1)
        rec = {
            "pair": a_ch + b_ch,
            "w_before": [before_ij, before_ji],
            "w_after": [after_ij, after_ji],
            "learned_edges": t.learned_edges[:12],
            "rank_of_partner_before": rank_before,
            "rank_of_partner_after": rank_after,
            "second_run_final": [c for c, _ in t2.final[:25] if c not in PUNCT],
        }
        novel.append(rec)
        print(f"    {a_ch}<->{b_ch}: 连接 {before_ij:.3f}->{after_ij:.3f} / "
              f"{before_ji:.3f}->{after_ji:.3f}; 单独激活{a_ch}时{b_ch}的名次 "
              f"{rank_before} -> {rank_after}")
        save_trace(f"07_novel_{a_ch}{b_ch}", t)
    R["experiment7_novel"] = novel

    with open(os.path.join(args.out, "experiments.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n结果已写入 {args.out}\\experiments.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
