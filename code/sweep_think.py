# -*- coding: utf-8 -*-
"""思考动力学的参数扫描: 找一组"既不发散也不死板、思考范围清楚"的参数。

评价指标:
  * 话题保留率 topic: 激活集合里还剩多少真正相关的字 (人工给的正例表)
  * 虚词占比 func: 的/是/不/了/和 ... 这类通用字占了多少 (越低越聚焦)
  * 规模 n: 稳定后的激活字数 (target_active 附近最好)
  * 区分度 sep: "苹果手机" 与 "苹果香蕉" 两组思考的重合度 (越低说明上下文越能改变思考)
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import CharBrain  # noqa: E402

MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "smoke.npz")

FUNC = set("的是不我了和有为最在也就都而及与被把对从到着过得地吗呢吧啊这那你他它我"
           "一二三四五六七八九十个年月日时分上下前后中大小多少好会很更还没只可以要"
           "，。、；：？！“”（）《》—…·")

TOPIC = {
    "苹果手机": set("苹果手机价格华为小米三星发布屏幕芯片电池旗舰机型销量市场产品用户"
                    "苹果公司科技系列版本功能支持新款上市处理器摄像头快充信号系统"),
    "苹果香蕉": set("苹果香蕉水果蔬菜葡萄西瓜草莓梨子果汁味道营养维生素吃甜酸新鲜"
                    "香蕉苹果市场价格产地进口国产热带"),
    "中国共产党": set("中国共产党中华人民共和国政府国家人民社会制度发展建设经济改革"
                      "领导政策会议中央委员会社会主义"),
    "人工智能": set("人工智能智能机器学习模型数据算法训练神经网络技术计算机科技"
                    "发展应用研究领域能力系统"),
}


def evaluate(brain: CharBrain, seeds: str, steps: int, topk: int = 40):
    tr = brain.think(seeds, steps=steps)
    final = [c for c, _ in tr.final[:topk]]
    n = tr.n_mean
    func = sum(1 for c in final if c in FUNC) / max(len(final), 1)
    topic = TOPIC.get(seeds)
    if topic:
        hit = sum(1 for c in final if c in topic) / max(len(final), 1)
    else:
        hit = float("nan")
    return {"n": n, "swing": tr.n_swing, "func": func, "topic": hit,
            "final": final, "theta_end": tr.theta_end}


def main() -> int:
    brain = CharBrain.load(MODEL)
    base = brain.p
    print(f"模型 {MODEL}: V={brain.V}")

    grid = []
    for alpha, sharp, out_soft, tgt in itertools.product(
            [0.0, 0.15, 0.3, 0.5], [1.5, 2.0, 3.0], [0.1, 0.25], [40, 80]):
        grid.append(dict(norm_alpha=alpha, sharp_pow=sharp, out_soft=out_soft,
                         target_active=tgt))
    mode = sys.argv[1] if len(sys.argv) > 1 else "sum"
    print(f"扫描 {len(grid)} 组 (drive_mode={mode}) ...")
    rows = []
    for gi, cfg in enumerate(grid, 1):
        p = base
        for k, v in cfg.items():
            setattr(p, k, v)
        p.drive_mode = mode
        recs = {}
        for seeds in TOPIC:
            recs[seeds] = evaluate(brain, seeds, steps=16)
        topic = np.nanmean([r["topic"] for r in recs.values()])
        func = np.nanmean([r["func"] for r in recs.values()])
        n = np.nanmean([r["n"] for r in recs.values()])
        swing = np.nanmean([r["swing"] for r in recs.values()])
        a = set(recs["苹果手机"]["final"])
        b = set(recs["苹果香蕉"]["final"])
        sep = len(a & b) / max(len(a | b), 1)
        rows.append((topic, func, n, swing, sep, cfg, recs))

    rows.sort(key=lambda r: (-r[0], r[1]))
    print(f"{'topic':>6} {'func':>6} {'n':>6} {'swing':>6} {'sep':>5}  cfg")
    for topic, func, n, swing, sep, cfg, _ in rows[:20]:
        print(f"{topic:6.3f} {func:6.3f} {n:6.0f} {swing:6.0f} {sep:5.2f}  "
              f"α={cfg['norm_alpha']} γ={cfg['sharp_pow']} soft={cfg['out_soft']} "
              f"target={cfg['target_active']}")
    best = rows[0]
    print("\n最佳配置的思考结果:")
    for seeds, r in best[6].items():
        print(f"  [{seeds}] n={r['n']:.0f} θ={r['theta_end']:.4f} 虚词占比={r['func']:.2f} "
              f"话题命中={r['topic']:.2f}")
        print("     " + " ".join(r["final"][:36]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
