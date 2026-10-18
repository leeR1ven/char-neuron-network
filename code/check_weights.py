# -*- coding: utf-8 -*-
"""快速体检: 新权重机制下, 关联是否合理、通用字是否被压住、定向性是否保持。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import CharBrain  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
b = CharBrain.load(os.path.join(HERE, "models", "char_brain.npz"))
print(f"模型 V={b.V}  连接={int((b.W>0).sum()):,}  "
      f"growth={b.p.growth} use_norm={b.p.use_norm}")
print("\n# 最强后继 (连接强度)")
for ch in "苹果手机月医芯诗虎医水火":
    i = b.char2id.get(ch)
    if i is None:
        continue
    row = b.W[i]
    idx = np.argsort(-row)[:7]
    print(f"  {ch} -> " + ", ".join(f"{b.chars[j]}{row[j]:.2f}" for j in idx if row[j] > 0))

g = np.argsort(-b.freq)[:200]
gen = float(np.mean([b.W[i][b.W[i] > 0].mean() for i in g]))
allw = float(b.W[b.W > 0].mean())
print(f"\n# 通用字出边平均权重 {gen:.3f} vs 全网平均 {allw:.3f} (比值 {gen/allw:.2f})")

print("\n# 有向性 (语序决定方向)")
for a, c in [("苹", "果"), ("手", "机"), ("医", "院"), ("月", "亮"), ("机", "器")]:
    i, j = b.char2id.get(a), b.char2id.get(c)
    if i is None or j is None:
        continue
    print(f"  {a}→{c}={b.W[i,j]:.3f}   {c}→{a}={b.W[j,i]:.3f}")

print("\n# 权重分布 (不再堆在 1.0)")
nz = b.W[b.W > 0]
for q in (10, 25, 50, 75, 90, 99, 99.9, 100):
    print(f"  {q:5.1f} 分位: {np.percentile(nz, q):.3f}")
print(f"  等于 1.0 的比例: {float((nz >= 0.999).mean())*100:.2f}%")
