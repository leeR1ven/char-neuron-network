# -*- coding: utf-8 -*-
"""大规模随机搜索: 把能调的参数都试一遍, 用自动指标挑出"聚焦但仍有内容"的组合。

权重侧参数 (要重建权重, 慢): 增长曲线 k_sat/c_ref、PMI 闸门、锐化 γ、归一化 α、top_k/c_min
动力学参数 (只改思考, 快): 激活规模、输出软度、衰减、增益控制、驱动方式、种子强度、
                            抑制增益、阈值跟进速度、思考步数

自动指标 (对一批种子求平均):
  topic   人工整理的话题命中率 (越高越"想到点子上")
  func    虚词占比 (越低越好)
  coh     激活集合内部平均关联度 (这些字彼此在语料里是否也常一起出现; 越高越"自洽")
  sep     不同上下文之间思考范围的重合度 (越低说明上下文越能改变思路)
  size    稳定后的激活字数 (贴近 target 说明阈值调控正常)
  swing   后半程激活数波动 (越小越稳)
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
from brain import CharBrain, Params, build_inhibition, build_weights  # noqa: E402
from rebuild import load_counts  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PUNCT = set("，。、；：？！“”‘’（）《》〈〉【】—…·～,.!?;:\"'()[]")
FUNC = set("的是不我了和有为最在也就都而及与被把对从到着过得地吗呢吧啊这那你他它"
           "一二三四五六七八九十个年月日时分上下前后中大多少好会很更还没只可以要")

TOPIC = {
    "苹果手机": set("苹果手机价格华为小米三星发布屏幕芯片电池旗舰机型销量市场产品用户"
                    "公司科技系列版本功能支持新款上市处理器摄像头快充信号系统厂商品牌"),
    "苹果香蕉": set("苹果香蕉水果蔬菜葡萄西瓜草莓梨汁味道营养维生素吃甜酸新鲜"
                    "市场产地进口国产热带食物"),
    "股票市场价格": set("股票价格市场涨跌指数行情投资交易证券基金买卖估值走势"
                        "上涨下跌涨停跌停成交量资金板块"),
    "医生给病人看病": set("医生病人看病医院治疗疾病药物手术检查诊断症状患者"
                          "健康护理门诊住院处方"),
    "北京上海广州": set("北京上海广州城市省市区县交通经济人口发展建设"
                        "滨海港口机场地铁都市繁华"),
    "今天天气不错我想出去走走": set("今天天气不错想出去走走阳光晴雨温度风冷热"
                                    "心情散步公园"),
    "人工智能技术": set("人工智能技术算法模型数据机器学习神经网络计算机"
                        "研究应用发展领域系统能力训练"),
    "汽车销售下降": set("汽车销售下降市场销量品牌车型价格优惠增长新能源"
                        "电动车燃油经销商"),
    "孩子学习成绩": set("孩子学习成绩考试分数老师学校家长教育作业课程"
                        "努力进步班级排名"),
    "足球比赛结果": set("足球比赛结果球队进球联赛冠军胜负积分教练球员"
                        "主场客场赛季"),
}
SEEDS = list(TOPIC)


def load_matrix(path):
    z = np.load(path, allow_pickle=False)
    V = len(z["chars"])
    C = np.zeros((V, V), dtype=np.int32)
    C[z["rows"], z["cols"]] = z["data"]
    return C, [str(c) for c in z["chars"]], z["freq"]


def evaluate(b: CharBrain, C: np.ndarray, c2i: dict, seeds, steps: int, topn: int = 35):
    topic, func, coh, size, swing = [], [], [], [], []
    sets = {}
    for s in seeds:
        tr = b.think(s, steps=steps)
        act = [c for c, _ in tr.final[:topn] if c not in PUNCT]
        sets[s] = set(act)
        topic.append(len(set(act) & TOPIC[s]) / max(len(TOPIC[s]), 1))
        func.append(sum(1 for c in act if c in FUNC) / max(len(act), 1))
        ids = [c2i[c] for c in act if c in c2i]
        if len(ids) >= 2:
            sub = C[np.ix_(ids, ids)].astype(np.float64)
            n = len(ids)
            off = (sub.sum() - np.trace(sub)) / (n * (n - 1))
            coh.append(float(np.log1p(off)))
        size.append(tr.n_mean)
        swing.append(tr.n_swing)
    # 上下文之间的区分度
    seps = []
    keys = list(sets)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, bb = sets[keys[i]], sets[keys[j]]
            seps.append(len(a & bb) / max(len(a | bb), 1))
    return {
        "topic": float(np.mean(topic)), "func": float(np.mean(func)),
        "coh": float(np.mean(coh)) if coh else 0.0,
        "sep": float(np.mean(seps)) if seps else 0.0,
        "size": float(np.mean(size)), "swing": float(np.mean(swing)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default=os.path.join(HERE, "models", "counts_char_brain.npz"))
    ap.add_argument("--weight-trials", type=int, default=24)
    ap.add_argument("--think-trials", type=int, default=120)
    ap.add_argument("--steps", type=int, default=18)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "sweep_all.json"))
    ap.add_argument("--seed", type=int, default=20261011)
    args = ap.parse_args()

    C, chars, freq = load_matrix(args.counts)
    c2i = {c: i for i, c in enumerate(chars)}
    rng = random.Random(args.seed)
    print(f"共现矩阵 V={len(chars)} 非零={int((C>0).sum()):,}", flush=True)

    # ---------- 1) 权重侧随机搜索 ----------
    wrows = []
    for t in range(args.weight_trials):
        wp = dict(
            growth=rng.choice(["log", "log", "sat"]),
            k_sat=rng.choice([1.0, 3.0, 8.0, 20.0]),
            c_ref=rng.choice([1000.0, 3000.0, 10000.0, 30000.0]),
            top_k=rng.choice([32, 64, 128]),
            c_min=rng.choice([2, 3, 8]),
            use_pmi=rng.random() < 0.8,
            pmi_temp=rng.choice([0.6, 1.2, 2.0]),
            pmi_floor=rng.choice([0.0, 0.12, 0.3]),
            pmi_pow=rng.choice([1.0, 1.4, 2.0]),
            sharp_pow=rng.choice([1.0, 1.5, 2.0, 3.0]),
            norm_alpha=rng.choice([0.0, 0.25, 0.5, 0.75, 1.0]),
            conf_k=0.0, prior_mode="off",
        )
        p = Params(**wp)
        # 权重侧的评估用一组中性的动力学参数
        p.target_active = 20
        p.out_soft = 0.15
        p.decay = 0.35
        p.gain_control = 0.02
        t0 = time.time()
        try:
            W, Wn, st, factors = build_weights(C, p, char_freq=freq)
        except Exception as e:  # noqa: BLE001
            print("  build ERR", type(e).__name__, str(e)[:60], wp)
            continue
        inh = build_inhibition(C, freq, p)
        b = CharBrain(chars, W, Wn, inh, p, {}, factors=factors)
        b.freq = freq
        m = evaluate(b, C, c2i, SEEDS, args.steps)
        m["weight_params"] = wp
        m["seconds"] = round(time.time() - t0, 1)
        wrows.append(m)
        print(f"  [W {t+1}/{args.weight_trials}] topic={m['topic']:.3f} func={m['func']:.3f} "
              f"coh={m['coh']:.2f} sep={m['sep']:.2f} n={m['size']:.0f} "
              f"γ={wp['sharp_pow']} α={wp['norm_alpha']} K={wp['k_sat']} "
              f"Cref={wp['c_ref']:.0f} topk={wp['top_k']} ({m['seconds']}s)", flush=True)

    # 每组权重下再随机试动力学参数
    trows = []
    for wi, m in enumerate(wrows):
        wp = m["weight_params"]
        p0 = Params(**wp)
        try:
            W, Wn, st, factors = build_weights(C, p0, char_freq=freq)
        except Exception:  # noqa: BLE001
            continue
        inh = build_inhibition(C, freq, p0)
        b = CharBrain(chars, W, Wn, inh, p0, {}, factors=factors)
        b.freq = freq
        k = max(1, args.think_trials // max(len(wrows), 1))
        for _ in range(k):
            tp = dict(
                target_active=rng.choice([10, 12, 15, 20, 25, 30, 40]),
                out_soft=rng.choice([0.05, 0.1, 0.15, 0.25, 0.4]),
                decay=rng.choice([0.1, 0.2, 0.3, 0.45, 0.6]),
                gain_control=rng.choice([0.0, 0.01, 0.02, 0.05, 0.1]),
                drive_mode=rng.choice(["sum", "sum", "lse"]),
                lse_beta=rng.choice([6.0, 12.0, 20.0]),
                seed_gain=rng.choice([0.6, 1.0, 1.5]),
                inh_gain=rng.choice([0.0, 0.3, 0.6, 1.0]),
                theta_lr=rng.choice([0.15, 0.25, 0.5]),
            )
            for kk, vv in tp.items():
                setattr(p0, kk, vv)
            p0.steps = args.steps
            mm = evaluate(b, C, c2i, SEEDS, args.steps)
            mm["weight_params"] = wp
            mm["think_params"] = tp
            trows.append(mm)

    trows.sort(key=lambda r: (-(r["topic"] - r["func"] * 0.5 - r["sep"] * 0.3
                                - r["swing"] / max(r["size"], 1) * 0.2)))
    print(f"\n{'topic':>6} {'func':>6} {'coh':>5} {'sep':>5} {'size':>5} {'swing':>6}  参数")
    for r in trows[:15]:
        tp = r["think_params"]
        wp = r["weight_params"]
        print(f"{r['topic']:6.3f} {r['func']:6.3f} {r['coh']:5.2f} {r['sep']:5.2f} "
              f"{r['size']:5.0f} {r['swing']:6.0f}  tgt={tp['target_active']} "
              f"soft={tp['out_soft']} decay={tp['decay']} gain={tp['gain_control']} "
              f"mode={tp['drive_mode']} β={tp['lse_beta']} seed={tp['seed_gain']} "
              f"inh={tp['inh_gain']} θlr={tp['theta_lr']} | γ={wp['sharp_pow']} "
              f"α={wp['norm_alpha']} K={wp['k_sat']} Cref={wp['c_ref']:.0f} "
              f"topk={wp['top_k']} pmiT={wp['pmi_temp']} pmiP={wp['pmi_pow']}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"weight_trials": wrows, "think_trials": trows}, f,
                  ensure_ascii=False, indent=2, default=float)
    print(f"\n结果写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
