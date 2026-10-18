# -*- coding: utf-8 -*-
"""打印"按方向"的召回结果。"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, "results", "recall_by_direction.json"), encoding="utf-8"))
for r in rows:
    print("=" * 76)
    print(f"【{r['tag']}】{r['sent']}")
    print(f"  正向(用句中先出现的概念当线索): 对方最高驱动 "
          f"{r['fwd_max'][0]:.2f} -> {r['fwd_max'][1]:.2f}  "
          f"带出 {len(r['fwd_hits'])} 字: {''.join(r['fwd_hits'])}")
    print(f"      思考: {''.join(r['fwd_content'][:28])}")
    print(f"  反向(用句中后出现的概念当线索): 带出 {len(r['bwd_hits'])} 字: "
          f"{''.join(r['bwd_hits'])}")
    print(f"      思考: {''.join(r['bwd_content'][:28])}")
