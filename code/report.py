# -*- coding: utf-8 -*-
"""生成中文实验报告 (Markdown): 语料、网络规模、机制、实测结果。

用法: python report.py --model models/char_brain.npz --results results --out 报告.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import CharBrain  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PUNCT = set("，。、；：？！“”‘’（）《》〈〉【】—…·～,.!?;:\"'()[]")


def human(n):
    for u in ("", "K", "M", "G", "T"):
        if abs(n) < 1000:
            return f"{n:.1f}{u}"
        n /= 1000.0
    return f"{n:.1f}P"


def pretty(cs, lim=None):
    cs = [c for c in cs if c not in PUNCT]
    if lim:
        cs = cs[:lim]
    return "".join(cs)


def corpus_stats() -> list[dict]:
    out = []
    raw = os.path.join(HERE, "corpus", "raw")
    for d in sorted(glob.glob(os.path.join(raw, "*"))):
        if not os.path.isdir(d):
            continue
        fs = [f for f in glob.glob(os.path.join(d, "**", "*"), recursive=True) if os.path.isfile(f)]
        out.append({"name": os.path.basename(d), "files": len(fs),
                    "bytes": sum(os.path.getsize(f) for f in fs)})
    for p in sorted(glob.glob(os.path.join(HERE, "corpus", "*.txt"))):
        b = os.path.basename(p)
        if "seen" in b or b.startswith("_"):
            continue
        out.append({"name": "网页抓取:" + b, "files": 1, "bytes": os.path.getsize(p)})
    clean = glob.glob(os.path.join(HERE, "corpus", "clean", "*.txt"))
    out.append({"name": "清洗后总量", "files": len(clean),
                "bytes": sum(os.path.getsize(f) for f in clean)})
    return out


def growth_table(p, counts=(1, 3, 10, 30, 100, 300, 1000, 3000, 10000)):
    rows = []
    for c in counts:
        if p.growth == "sat":
            w = c / (c + p.k_sat)
        else:
            w = min(np.log1p(c / p.k_sat) / np.log1p(p.c_ref / p.k_sat), 1.0)
        rows.append((c, float(w)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "models", "char_brain.npz"))
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--out", default=os.path.join(HERE, "结果", "字神经元网络_实验报告.md"))
    args = ap.parse_args()

    b = CharBrain.load(args.model)
    meta = b.meta
    exp_path = os.path.join(args.results, "experiments.json")
    E = {}
    if os.path.exists(exp_path):
        with open(exp_path, encoding="utf-8") as f:
            E = json.load(f)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    L: list[str] = []
    A = L.append
    A(f"# 字神经元关联网络 —— 训练与思考实验报告\n")
    A(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}　模型: `{os.path.basename(args.model)}`\n")

    # ---------------- 一、语料
    A("## 一、语料\n")
    A("| 来源 | 文件数 | 体积 |")
    A("|---|---:|---:|")
    for c in corpus_stats():
        A(f"| {c['name']} | {c['files']} | {human(c['bytes'])}B |")
    A("")
    A(f"本次训练实际读入 **{meta.get('corpus_chars_read', 0):,}** 字，"
      f"用于统计的共现对 **{meta.get('bigrams', 0):,}** 个；"
      f"语料为多来源混合（新闻网页、百科、论坛问答、小说、诗词、通用网页），"
      f"按各来源体积比例抽样，避免单一来源主导。\n")

    # ---------------- 二、网络
    A("## 二、神经元与连接\n")
    A(f"- 神经元（一个字 = 一个神经元）: **{b.V:,}** 个，字频门槛 {meta.get('min_freq')} 次")
    A(f"- 有向连接: **{meta.get('edges', 0):,}** 条（每个字最多保留 {meta.get('top_k')} 条出边，"
      f"共现次数 ≥ {meta.get('c_min')}）")
    A(f"- 每个字配一个抑制神经元，抑制连接 **{meta.get('inh_edges', 0):,}** 条")
    A(f"- 稀疏度: {(meta.get('edges',0)/max(b.V*b.V,1))*100:.3f}%（全连接矩阵的占比）")
    A("")
    A("### 有向性: 顺序决定方向\n")
    A("| 前字 | 最强后继（连接强度） |")
    A("|---|---|")
    for ch in "苹果手机中国的水火":
        i = b.char2id.get(ch)
        if i is None:
            continue
        row = b.W[i]
        idx = np.argsort(-row)[:5]
        top = "、".join(f"{b.chars[j]} {row[j]:.2f}" for j in idx if row[j] > 0)
        A(f"| {ch} | {top} |")
    A("")
    for pair in ("苹果", "果苹", "手机", "机手"):
        i, j = b.char2id.get(pair[0]), b.char2id.get(pair[1])
        if i is None or j is None:
            continue
        A(f"- `{pair[0]}→{pair[1]}` 强度 **{b.W[i, j]:.3f}**")
    A("")

    # ---------------- 三、权重增长机制
    A("## 三、权重增长机制（越涨越难涨，少见连接不被压死）\n")
    A(f"连接强度 w = log(1 + c/K) / log(1 + C_ref/K)，K={b.p.k_sat}，C_ref={b.p.c_ref:.0f}，"
      f"再乘 PMI 特异性闸门（抑制“的→是”这类万金油搭配）。\n")
    A("| 共现次数 c | 1 | 3 | 10 | 30 | 100 | 300 | 1000 | 3000 | 10000 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    gt = growth_table(b.p)
    A("| 连接强度 w | " + " | ".join(f"{w:.3f}" for _, w in gt) + " |")
    A("")
    A("边际增幅约为 1/c：出现 3 次就能拿到约 0.2 的强度，之后每多出现一次涨得越来越少，"
      "因此冷门搭配不会被常见搭配永远压制。\n")

    # ---------------- 四、思考机制
    A("## 四、思考（传播激活）机制\n")
    A("1. **传播**：`drive_j = Σ_i a_i · Ŵ[i,j] /(1+γ·Σa)`，"
      "Ŵ 在行、列两侧都做了归一化——出边多的常见字被摊薄，"
      "谁都想连的万金油字也被摊薄，于是少见字的专属连接照样推得动。")
    A("2. **抑制神经元**：每个字一个抑制神经元，"
      "“很常见但和它几乎不搭”的字会给它负权重，抵消一部分正驱动。")
    A(f"3. **整体调控阈值**：所有神经元共用同一个阈值 θ，每步按“让大约 "
      f"{b.p.target_active} 个字越过阈值”反推 θ 并平滑跟进——"
      f"激活太少就整体降阈值，太多就整体升阈值。阈值不固定在单个神经元上。")
    A(f"4. **实时学习**：思考过程中同时活跃的字按 `Δw = η·a_i·a_j·(1-w/w_max)^pow` 加强连接"
      f"（η={b.p.learn_eta}），同样越涨越难涨；很少一起出现的字被一起激活后能长出新的通路。\n")

    # ---------------- 五、实验
    A("## 五、实验与结果\n")
    e1 = E.get("experiment1_threshold")
    if e1:
        A("### 1. 阈值整体调控\n")
        A(f"种子 `{e1['seeds']}`，目标激活数 {e1['target_active']}：\n")
        A(f"- 激活字数轨迹: `{e1['n_active_hist']}`")
        A(f"- 阈值 θ 轨迹: `{[round(x,4) for x in e1['theta_hist']]}`")
        A(f"- 稳定后平均激活 **{e1['n_mean']:.1f}** 个字（目标 {e1['target_active']}），"
          f"后半段波动 {e1['n_swing']:.0f}\n")
    e2 = E.get("experiment2_context")
    if e2:
        A("### 2. 激活不同上下文，思考范围随之改变\n")
        A("| 激活的字 | 后续激活（前 30 个，去掉标点） |")
        A("|---|---|")
        for k, v in e2["sets"].items():
            A(f"| {k} | {pretty(v, 30)} |")
        A("")
        A(f"`苹果手机` 与 `苹果香蕉` 两组思考的 Jaccard 重合度 "
          f"**{e2['jaccard_手机_vs_香蕉']:.3f}**（越低说明上下文越能决定思考方向）。\n")
        A(f"- 共同激活: {pretty(e2['共同'], 26)}")
        A(f"- 只在“苹果手机”下出现: {pretty(e2['只在手机出现'], 26)}")
        A(f"- 只在“苹果香蕉”下出现: {pretty(e2['只在香蕉出现'], 26)}\n")
    e3 = E.get("experiment3_random")
    if e3:
        A("### 3. 随机激活几个字\n")
        for r in e3:
            A(f"- 随机种子 `{r['seeds']}`（{r['n_mean']:.0f} 个字被激活）→ {pretty(r['final'], 30)}")
        A("")
    e4 = E.get("experiment4_sentence")
    if e4:
        A("### 4. 激活一整句话\n")
        for r in e4:
            A(f"- `{r['seeds']}`（θ={r['theta_end']:.3f}）")
            A(f"  - 最终激活: {pretty(r['final'], 36)}")
            A(f"  - 按激活先后顺序: {pretty(r['order'], 36)}")
        A("")
    e5 = E.get("experiment5_inhibition")
    if e5:
        A("### 5. 抑制神经元 开/关 对比\n")
        A("| 抑制 | 平均激活数 | 通用虚词占比 | 最终激活（前 24） |")
        A("|---|---:|---:|---|")
        for k, v in e5.items():
            A(f"| {k} | {v['n_mean']:.0f} | {v['func_ratio']:.2f} | {pretty(v['final'], 24)} |")
        A("")
    e6 = E.get("experiment6_rare")
    if e6:
        A("### 6. 少见连接保护（冷门字能不能被激活）\n")
        A(f"在 {len(e6['pairs'])} 个“冷门字→专属搭档”样本中，单独激活冷门字后搭档被激活的比例 "
          f"**{e6['hit_rate']*100:.0f}%**。\n")
        A("| 冷门字(字频) | 专属搭档 | 连接强度 | 单独激活后是否带出搭档 |")
        A("|---|---|---:|---|")
        for d in e6["demos"]:
            A(f"| {d['rare']}({d['freq']}) | {d['partner']} | {d['w']:.3f} | "
              f"{'是' if d['partner_activated'] else '否'} |")
        A("")
    e7 = E.get("experiment7_novel")
    if e7:
        A("### 7. 新思想：本来不搭界的字一起激活后长出连接\n")
        A("| 字对 | 学习前连接 | 一起激活一轮后 | 单独激活前者时后者的名次 |")
        A("|---|---|---|---|")
        for r in e7:
            A(f"| {r['pair']} | {r['w_before'][0]:.3f} / {r['w_before'][1]:.3f} | "
              f"{r['w_after'][0]:.3f} / {r['w_after'][1]:.3f} | "
              f"{r['rank_of_partner_before']} → {r['rank_of_partner_after']} |")
        A("")
        A("说明: 连接按“越涨越难涨”的规则增长，一次共激活只会小步加强，"
          "反复一起出现才会长成稳定通路——这就是“新思想”的物理形式。\n")

    # ---------------- 六、参数
    A("## 六、当前参数\n")
    A("```")
    for k, v in sorted(vars(b.p).items()):
        A(f"{k} = {v}")
    A("```")
    A("")
    A("> 所有数据、模型与代码均在 `F:\\harness关联神经元架构\\字神经元\\` 下："
      "`corpus/` 语料、`models/` 模型、`results/` 实验轨迹。")

    text = "\n".join(L)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"报告已写入 {args.out} ({len(text)} 字)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
