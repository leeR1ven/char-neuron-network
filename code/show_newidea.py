# -*- coding: utf-8 -*-
"""把 newidea_formal.json 汇总成一张人看的表。"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(HERE, "results", "newidea_formal.json")
data = json.load(open(p, encoding="utf-8"))
for c in data:
    print("=" * 74)
    print(f"【{c['tag']}】学习句: {c['sentence']}")
    print(f"  连接: {c['A']}→{c['B']} {c['w_before'][0]:.3f}→{c['w_after'][0]:.3f} | "
          f"{c['B']}→{c['A']} {c['w_before'][1]:.3f}→{c['w_after'][1]:.3f}")
    print("  跨簇连接: " + ", ".join(f"{x}→{y}({w:.2f})" for x, y, w in c["cross_edges"][:8]))
    print(f"  线索A「{c['cue_A']}」")
    print(f"     基线带出B簇 {len(c['baseline_A_hits'])} 字 → 学习后 {len(c['after_A_hits'])} 字 "
          f"[{''.join(c['after_A_hits'])}]  B本体被想起={c['after_A_rankB'] > 0}")
    print(f"     学习后思考: {''.join(c['after_A_content'][:28])}")
    print(f"  线索B「{c['cue_B']}」")
    print(f"     基线带出A簇 {len(c['baseline_B_hits'])} 字 → 学习后 {len(c['after_B_hits'])} 字 "
          f"[{''.join(c['after_B_hits'])}]  A本体被想起={c['after_B_rankA'] > 0}")
    print(f"     学习后思考: {''.join(c['after_B_content'][:28])}")
    print(f"  学习句本身再次思考: {''.join(c['learn_sentence_content'][:28])}")
