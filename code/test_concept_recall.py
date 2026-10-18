# -*- coding: utf-8 -*-
"""概念级实验: 说一句话把两个概念绑在一起, 之后只谈其中一个概念, 看能不能带出另一个概念。

概念 = 网络自己点亮的一群字 (用一句只谈该概念的话激活出来的集合), 不是一个字。

测量 (都用"整簇"口径):
  * 重叠   : 线索激活集合里, 落在另一个概念簇里的字数
  * 质量   : 另一个概念簇上聚集的总激活量 Σ a_j
  * 对照   : 随机取同等大小的簇 200 次, 得到重叠/质量的随机基线 (说明是不是"显著")
  * 方向   : 学习句里谁先出现, 谁就获得指向对方的连接 (顺序决定方向)

用法: python test_concept_recall.py [--eta 0.05] [--rounds 5]
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

# (标签, 学习句(较长, 含两个相距较远的概念), 只谈概念A的话, 只谈概念B的话)
CASES = [
    ("医学 × 芯片",
     "医院的医生正在研究一种新型芯片，希望用它帮病人更早发现疾病并且治好病",
     "医院的医生正在给病人治疗", "芯片的电路板和材料"),
    ("野兽 × 金融",
     "动物园里的老虎最近出了名，很多人专门跑到银行存款，想给它买肉吃",
     "动物园里的老虎和狮子", "银行存款和利息"),
    ("天体 × 水果",
     "科学家说月球表面的土壤里也许能种出苹果和蔬菜，让宇航员吃上新鲜水果",
     "月球表面和星星", "苹果树和水果"),
    ("机器 × 文学",
     "工厂里的机器人开始学习读诗写文章，还能给孩子们讲古代诗人的故事",
     "机器和自动化生产", "诗歌和文学作品"),
    ("医院 × 战争",
     "战争爆发以后，医院里送来了很多受伤的士兵，医生们连夜给他们做手术",
     "医院和医生的日常治疗", "战争和军队的冲突"),
]


PROBES = [
    "今天天气不错我想出去走走", "股票市场价格波动很大", "孩子们在操场上踢足球",
    "政府发布了新的政策", "我在家里做饭洗衣服", "这部电影的演员很有名",
    "路上堵车迟到了半小时", "他每天早上跑步锻炼身体", "网上买东西便宜又方便",
    "小区门口的超市关门了", "这首歌的旋律很好听", "会议室里坐满了人",
]


def content_chars(b: CharBrain, k: int = 300) -> set:
    """实义字集合: 排除语料里最高频的 k 个"胶水字"(的/和/在/是...)与标点。"""
    order = np.argsort(-b.freq)
    glue = {b.chars[i] for i in order[:k]}
    return {c for c in b.chars if c not in glue and c not in PUNCT}


def activation_vector(b: CharBrain, cue: str, steps: int) -> np.ndarray:
    """跑完一整轮, 返回最终激活向量。"""
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
    """通用背景: 一堆不相干的句子平均下来, 每个字被点亮的程度。"""
    acc = np.zeros(b.V, dtype=np.float64)
    for s in PROBES:
        acc += activation_vector(b, s, steps)
    return (acc / len(PROBES)).astype(np.float32)


def concept_cluster(b: CharBrain, cue: str, bg: np.ndarray, keep: set,
                    steps: int, topn: int = 20):
    """概念簇 = 该线索比"通用背景"明显更亮的实义字 (区分度口径)。"""
    a = activation_vector(b, cue, steps)
    d = a - bg
    ids = [i for i in range(b.V) if b.chars[i] in keep]
    ids.sort(key=lambda i: -d[i])
    picked = [i for i in ids[:topn] if d[i] > 0.05]
    return {b.chars[i] for i in picked}, a, d


def mass_on(b: CharBrain, a_vec, cluster: set) -> float:
    ids = [b.char2id[c] for c in cluster if c in b.char2id]
    return float(a_vec[ids].sum()) if ids else 0.0


def control_stats(b: CharBrain, cluster_size: int, active_ids, a_vec, trials: int = 200,
                  seed: int = 7):
    """随机同等大小簇的重叠/质量基线。"""
    rng = random.Random(seed)
    pool = [i for i in range(b.V) if b.chars[i] not in PUNCT]
    ov, ms = [], []
    active_set = set(active_ids.tolist())
    for _ in range(trials):
        pick = rng.sample(pool, cluster_size)
        ov.append(sum(1 for i in pick if i in active_set))
        ms.append(float(a_vec[pick].sum()))
    return float(np.mean(ov)), float(np.percentile(ov, 95)), float(np.mean(ms))


def eval_cue(b: CharBrain, cue: str, other_cluster: set, steps: int, keep: set | None = None,
             bg: np.ndarray | None = None):
    a = activation_vector(b, cue, steps)
    if bg is None:
        bg = np.zeros(b.V, dtype=np.float32)
    d = a - bg
    names = [b.chars[i] for i in np.argsort(-d) if b.chars[i] in (keep or set(b.chars))][:25]
    hits = [c for c in names if c in other_cluster]
    ids = [b.char2id[c] for c in other_cluster if c in b.char2id]
    mass = float(np.maximum(d[ids], 0).sum()) if ids else 0.0
    active_ids = np.nonzero(a > b.p.act_eps)[0]
    return {"content": names, "hits": hits, "mass": mass, "a": a, "d": d,
            "n_active": int(len(active_ids)), "active_ids": active_ids}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "models", "char_brain.npz"))
    ap.add_argument("--eta", type=float, default=None, help="默认用模型自带的学习步长")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=24, help="读线索句用的步数")
    ap.add_argument("--gap-decay", type=float, default=None,
                    help="学习时先后间隔的折扣; 1.0 = 不打折 (完全按顺序)")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "concept_recall.json"))
    args = ap.parse_args()

    base = CharBrain.load(args.model)
    base.p.steps = args.steps
    print(f"模型 V={base.V}  学习步长 η={args.eta if args.eta else base.p.learn_eta}"
          f"  重复 {args.rounds} 遍  每遍 {args.steps} 步\n")

    report = []
    keep = content_chars(base)
    bg = background(base, args.steps)
    print(f"通用背景已估计 ({len(PROBES)} 句), 实义字 {len(keep)} 个\n")
    for tag, sent, cue_a, cue_b in CASES:
        print("=" * 78)
        print(f"【{tag}】绑定句: 「{sent}」")
        cluster_a, _, _ = concept_cluster(base, cue_a, bg, keep, args.steps)
        cluster_b, _, _ = concept_cluster(base, cue_b, bg, keep, args.steps)
        print(f"  概念A簇({len(cluster_a)}字, 由「{cue_a}」点亮): {''.join(sorted(cluster_a))}")
        print(f"  概念B簇({len(cluster_b)}字, 由「{cue_b}」点亮): {''.join(sorted(cluster_b))}")

        base_a = eval_cue(base, cue_a, cluster_b, args.steps, keep, bg)
        base_b = eval_cue(base, cue_b, cluster_a, args.steps, keep, bg)
        ctrl_ov_a, ctrl_95_a, ctrl_m_a = control_stats(base, len(cluster_b),
                                                       base_a["active_ids"], base_a["a"])
        print(f"  [基线] 只谈A: B簇重叠 {len(base_a['hits'])} 字, B簇质量 {base_a['mass']:.3f} "
              f"(随机同规模簇: 重叠均值 {ctrl_ov_a:.2f} / 95分位 {ctrl_95_a:.0f}, 质量 {ctrl_m_a:.3f})")

        # ---- 绑定学习: 反复读这句话 (读完整句需要足够步数: 按语序逐字读入) ----
        b = copy.deepcopy(base)
        if args.eta:
            b.p.learn_eta = args.eta
        if args.gap_decay is not None:
            b.p.learn_gap_decay = args.gap_decay
        steps_learn = max(args.steps, len(sent) + 6)
        i_a = [b.char2id[c] for c in cluster_a if c in b.char2id]
        i_b = [b.char2id[c] for c in cluster_b if c in b.char2id]
        W_cross0 = float(b.W[np.ix_(i_a, i_b)].sum())
        for _ in range(args.rounds):
            b.think(sent, learn=True, steps=steps_learn, sequence=True)
        W_cross1 = float(b.W[np.ix_(i_a, i_b)].sum())
        n_cross = int((b.W[np.ix_(i_a, i_b)] > 0.1).sum())
        print(f"  绑定学习{args.rounds}遍 (每遍 {steps_learn} 步): "
              f"两簇之间连接总强度 {W_cross0:.2f} → {W_cross1:.2f}, "
              f"其中 >0.1 的连接 {n_cross} 条")

        after_a = eval_cue(b, cue_a, cluster_b, args.steps, keep, bg)
        after_b = eval_cue(b, cue_b, cluster_a, args.steps, keep, bg)
        ctrl_ov_a2, ctrl_95_a2, ctrl_m_a2 = control_stats(b, len(cluster_b),
                                                          after_a["active_ids"], after_a["a"])
        ctrl_ov_b2, ctrl_95_b2, ctrl_m_b2 = control_stats(b, len(cluster_a),
                                                          after_b["active_ids"], after_b["a"])
        print(f"  [学习后] 只谈A「{cue_a}」")
        print(f"      带出B簇 {len(after_a['hits'])} 字: {''.join(after_a['hits'])} "
              f"| B簇质量 {after_a['mass']:.3f} (随机基线 {ctrl_m_a2:.3f})")
        print(f"      思考内容: {''.join(after_a['content'])}")
        print(f"  [学习后] 只谈B「{cue_b}」")
        print(f"      带出A簇 {len(after_b['hits'])} 字: {''.join(after_b['hits'])} "
              f"| A簇质量 {after_b['mass']:.3f} (随机基线 {ctrl_m_b2:.3f})")
        print(f"      思考内容: {''.join(after_b['content'])}")

        report.append({
            "tag": tag, "sentence": sent, "cue_A": cue_a, "cue_B": cue_b,
            "cluster_A": sorted(cluster_a), "cluster_B": sorted(cluster_b),
            "cross_strength": [W_cross0, W_cross1], "cross_edges_gt01": n_cross,
            "baseline_A_hits": base_a["hits"], "baseline_A_mass": base_a["mass"],
            "control_A": [ctrl_ov_a, ctrl_95_a, ctrl_m_a],
            "after_A_hits": after_a["hits"], "after_A_mass": after_a["mass"],
            "after_A_control": [ctrl_ov_a2, ctrl_95_a2, ctrl_m_a2],
            "after_A_content": after_a["content"],
            "after_B_hits": after_b["hits"], "after_B_mass": after_b["mass"],
            "after_B_control": [ctrl_ov_b2, ctrl_95_b2, ctrl_m_b2],
            "after_B_content": after_b["content"],
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n结果写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
