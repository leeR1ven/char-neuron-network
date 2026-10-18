# -*- coding: utf-8 -*-
"""诊断: 给定种子字, 看每个候选字的驱动由谁贡献, 判断权重设计是否合理。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import CharBrain  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "smoke.npz")
SEEDS = sys.argv[2] if len(sys.argv) > 2 else "苹果手机"
TOP = int(sys.argv[3]) if len(sys.argv) > 3 else 25

b = CharBrain.load(MODEL)
print(f"模型 {MODEL}: V={b.V}, 连接={int((b.W>0).sum()):,}, 抑制连接={len(b.inh_idx):,}")
print(f"meta: {b.meta}")

ext = b.seed_vector(SEEDS)
ids = np.nonzero(ext > 0)[0]
print(f"种子: {[b.chars[i] for i in ids]}")

drive = ext @ b.Wn
order = np.argsort(-drive)[:TOP]
print("\n# 第一步驱动最强的字 (种子直接推起来的):")
for j in order:
    if drive[j] <= 0:
        continue
    contrib = [(b.chars[i], float(ext[i] * b.Wn[i, j]), float(b.W[i, j])) for i in ids
               if b.Wn[i, j] > 0]
    contrib.sort(key=lambda x: -x[1])
    src = ", ".join(f"{c}:传播{w:.4f}(强度{s:.2f})" for c, w, s in contrib[:3])
    print(f"  {b.chars[j]} 驱动={drive[j]:.4f}  ← {src}")

print("\n# 各自的出边有多少条 / 行和 (衡量'万金油'程度):")
for j in order[:12]:
    i = int(j)
    n_out = int((b.W[i] > 0).sum())
    n_in = int((b.W[:, i] > 0).sum())
    print(f"  {b.chars[i]}: 出边{n_out:3d} 行强度合计{b.W[i].sum():6.2f} "
          f"入边{n_in:5d} 列强度合计{b.W[:, i].sum():8.2f} "
          f"传播权重最大{b.Wn[i].max():.4f}")

print("\n# 反向检查: 常用虚词在传播矩阵里的量级")
for ch in "的是在和了也我你他。，":
    i = b.char2id.get(ch)
    if i is None:
        continue
    print(f"  {ch}: 出边{int((b.W[i]>0).sum()):3d} 传入权重合计={b.Wn[:, i].sum():.4f} "
          f"最大入权重={b.Wn[:, i].max():.5f}")
