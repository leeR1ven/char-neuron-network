# -*- coding: utf-8 -*-
"""概念级实验 (v2): 两个概念绑定 -> 只提一个概念 -> 看另一个概念有没有被想起来。

概念的定义: 线索句里的**实义字** + 它们的强关联字 (连接强度 ≥ 0.6, 每个字取前 8 个)。
  —— 这样"医学"= {医,院,病,治,疗,生,...} + {患,药,诊,护,师,科,...}, 干净可控;
  —— 不会把"和蔼/和睦/尴尬/佼"这种由虚词带出来的杂字算进来。

测量:
  * 重叠: 只提A时, 输出里出现多少个B概念的字 (随机同规模簇作为对照)
  * 质量: B概念上聚集的"相对通用背景多出来的"激活量
  * 方向: 学习句里谁先出现, 谁就得到指向对方的连接
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import CharBrain  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PUNCT = set("，。、；：？！“”‘’（）《》〈〉【】—…·～,.!?;:\"'()[]")
GLUE_EXTRA = set("的和在是了也都很就而及与被把对从到着过得地这那你他它我上下前后里会要能可以还但没有多个"
                 "一二三四五六七八九十个年月日时大小好更没只于其并且因所以如果虽然然而")

PROBES = [
    "今天天气不错我想出去走走", "股票市场价格波动很大", "孩子们在操场上踢足球",
    "政府发布了新的政策", "我在家里做饭洗衣服", "这部电影的演员很有名",
    "路上堵车迟到了半小时", "他每天早上跑步锻炼身体", "网上买东西便宜又方便",
    "小区门口的超市关门了", "这首歌的旋律很好听", "会议室里坐满了人",
]

# (标签, 长句(含两个相距较远的概念), A概念代表字, B概念代表字)
CASES = [
    ("医学 × 芯片", "医院的医生正在研究一种新型芯片，希望用它帮病人更早发现疾病并且治好病",
     "医院生病治疗", "芯片电路板材料"),
    ("野兽 × 金融", "动物园里的老虎最近出了名，很多人专门跑到银行存款，想给它买肉吃",
     "动物园老虎狮子", "银行存款利息"),
    ("天体 × 水果", "科学家说月球表面的土壤里也许能种出苹果和蔬菜，让宇航员吃上新鲜水果",
     "月球表面星星", "苹果树水果"),
    ("机器 × 文学", "工厂里的机器人开始学习读诗写文章，还能给孩子们讲古代诗人的故事",
     "机器工厂生产", "诗歌文学文章"),
    ("医院 × 战争", "战争爆发以后，医院里送来了很多受伤的士兵，医生们连夜给他们做手术",
     "医院医生治疗", "战争军队士兵"),
]


def activation_vector(b: CharBrain, cue: str, steps: int) -> np.ndarray:
    p = b.p
    ext = b.seed_vector(cue)
    seed_ids = np.nonzero(ext > 0)[0]
    a = ext.copy()
    d0 = ext @ b.Wn
    theta = float(np.clip(np.quantile(d0[d0 > 0], max(0.0, 1 - p.target_active / b.V)) * 0.6,
                          p.theta_min, p.theta_max)) if (d0 > 0).any() else p.theta_init
    k = min(p.target_active, b.V)
    for _ in range(steps):
        drive = a @ b.Wn
        s = float(a.sum())
        if p.gain_control > 0:
            drive = drive / (1.0 + p.gain_control * s)
        if p.use_inhibition and len(b.inh_idx):
            drive = drive - p.inh_gain * b._inhibition(a)
        pot = p.decay * a + drive + ext * p.seed_gain
        ts = float(np.partition(pot, b.V - k)[b.V - k]) - p.act_eps * p.out_soft
        theta = float(np.clip((1 - p.theta_lr) * theta + p.theta_lr * ts, p.theta_min, p.theta_max))
        a = np.clip((pot - theta) / p.out_soft, 0.0, 1.0)
        a[seed_ids] = np.maximum(a[seed_ids], ext[seed_ids])
    return a


def background(b: CharBrain, steps: int) -> np.ndarray:
    acc = np.zeros(b.V, dtype=np.float64)
    for s in PROBES:
        acc += activation_vector(b, s, steps)
    return (acc / len(PROBES)).astype(np.float32)


def concept_chars(b: CharBrain, core: str, per: int = 8, w_thresh: float = 0.6) -> set:
    """概念 = 代表字 + 它们的强关联字 (连接强度 ≥ w_thresh)。"""
    out = set()
    for ch in core:
        if ch not in b.char2id or ch in PUNCT:
            continue
        out.add(ch)
        row = b.W[b.char2id[ch]]
        idx = np.argsort(-row)[:per]
        for j in idx:
            if row[j] >= w_thresh and b.chars[j] not in PUNCT:
                out.add(b.chars[j])
    return {c for c in out if c not in GLUE_EXTRA}


def eval_cue(b: CharBrain, cue: str, other: set, steps: int, bg: np.ndarray, topn: int = 30):
    a = activation_vector(b, cue, steps)
    d = a - bg
    order = np.argsort(-d)
    names = [b.chars[i] for i in order if b.chars[i] not in PUNCT][:topn]
    hits = [c for c in names if c in other]
    ids = [b.char2id[c] for c in other if c in b.char2id]
    mass = float(np.maximum(d[ids], 0).sum()) if ids else 0.0
    return {"content": names, "hits": hits, "mass": mass, "a": a, "d": d}


def control(b: CharBrain, size: int, d: np.ndarray, trials: int = 300, seed: int = 11):
    rng = random.Random(seed)
    pool = [i for i in range(b.V) if b.chars[i] not in PUNCT]
    ms = []
    for _ in range(trials):
        pick = rng.sample(pool, size)
        ms.append(float(np.maximum(d[pick], 0).sum()))
    return float(np.mean(ms)), float(np.percentile(ms, 95))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "models", "char_brain.npz"))
    ap.add_argument("--eta", type=float, default=None)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--gap-decay", type=float, default=None)
    ap.add_argument("--w-thresh", type=float, default=0.6)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "concept_recall.json"))
    args = ap.parse_args()

    base = CharBrain.load(args.model)
    base.p.steps = args.steps
    print(f"模型 V={base.V}  η={args.eta if args.eta else base.p.learn_eta}  "
          f"重复{args.rounds}遍\n")
    bg = background(base, args.steps)
    report = []
    for tag, sent, core_a, core_b in CASES:
        print("=" * 80)
        print(f"【{tag}】绑定句({len(sent)}字): 「{sent}」")
        cA = concept_chars(base, core_a, w_thresh=args.w_thresh)
        cB = concept_chars(base, core_b, w_thresh=args.w_thresh)
        print(f"  概念A({len(cA)}字): {''.join(sorted(cA))}")
        print(f"  概念B({len(cB)}字): {''.join(sorted(cB))}")

        cue_a = "".join(core_a)
        base_a = eval_cue(base, cue_a, cB, args.steps, bg)
        m_ctrl, m95 = control(base, len(cB), base_a["d"])
        print(f"  [基线] 只说A「{cue_a}」→ 带出B概念 {len(base_a['hits'])} 字 "
              f"{''.join(base_a['hits'])} | B质量 {base_a['mass']:.2f} "
              f"(随机同规模 {m_ctrl:.2f}, 95分位 {m95:.2f})")

        # ---- 绑定学习 ----
        b = copy.deepcopy(base)
        if args.eta:
            b.p.learn_eta = args.eta
        if args.gap_decay is not None:
            b.p.learn_gap_decay = args.gap_decay
        steps_learn = max(args.steps, len(sent) + 6)
        ia = [b.char2id[c] for c in cA if c in b.char2id]
        ib = [b.char2id[c] for c in cB if c in b.char2id]
        w0 = float(b.W[np.ix_(ia, ib)].sum())
        for _ in range(args.rounds):
            b.think(sent, learn=True, steps=steps_learn, sequence=True)
        w1 = float(b.W[np.ix_(ia, ib)].sum())
        n_cross = int((b.W[np.ix_(ia, ib)] > 0.1).sum())
        top_cross = sorted(((b.chars[x], b.chars[y], float(b.W[x, y]))
                            for x in ia for y in ib if b.W[x, y] > 0.15),
                           key=lambda e: -e[2])[:10]
        print(f"  绑定{args.rounds}遍({steps_learn}步/遍): A→B 连接总强度 {w0:.2f} → {w1:.2f}, "
              f">0.1 的 {n_cross} 条")
        print("     最强跨概念连接: " + ", ".join(f"{x}→{y}({w:.2f})" for x, y, w in top_cross))

        after_a = eval_cue(b, cue_a, cB, args.steps, bg)
        cue_b = "".join(core_b)
        after_b = eval_cue(b, cue_b, cA, args.steps, bg)
        m_ctrl2, m952 = control(b, len(cB), after_a["d"])
        print(f"  [学习后] 只说A「{cue_a}」→ 带出B概念 {len(after_a['hits'])} 字 "
              f"{''.join(after_a['hits'])} | B质量 {after_a['mass']:.2f} "
              f"(随机 {m_ctrl2:.2f}, 95分位 {m952:.2f})")
        print(f"     思考: {''.join(after_a['content'])}")
        print(f"  [学习后] 只说B「{cue_b}」→ 带出A概念 {len(after_b['hits'])} 字 "
              f"{''.join(after_b['hits'])} | A质量 {after_b['mass']:.2f}")
        print(f"     思考: {''.join(after_b['content'])}")

        report.append({
            "tag": tag, "sentence": sent, "core_A": core_a, "core_B": core_b,
            "cluster_A": sorted(cA), "cluster_B": sorted(cB),
            "cross_strength": [w0, w1], "cross_gt01": n_cross,
            "top_cross_edges": top_cross,
            "baseline_hits": base_a["hits"], "baseline_mass": base_a["mass"],
            "baseline_control": [m_ctrl, m95],
            "after_hits": after_a["hits"], "after_mass": after_a["mass"],
            "after_control": [m_ctrl2, m952],
            "after_content_A": after_a["content"], "after_content_B": after_b["content"],
            "after_hits_B": after_b["hits"], "after_mass_B": after_b["mass"],
        })
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n结果写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
