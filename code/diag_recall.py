# -*- coding: utf-8 -*-
"""诊断: 一条新学到的连接要多强, 才能在"只提一个概念"时把另一个概念想起来?

衡量的是传播权重 Ŵ (真正决定能不能越过阈值的东西):
    Ŵ[i,j] = W[i,j]^γ / (row_i^α · col_eff_j^(1-α))
于是"要想起 j", 需要 Ŵ[i,j] ≥ 当前阈值 θ 附近的量级。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import CharBrain  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
b = CharBrain.load(os.path.join(HERE, "models", "char_brain.npz"))
f = b.factors
print(f"γ={f['gamma']} α={f['alpha']}  V={b.V}")

print("\n# 种子字的出边强度分布与最强后继的传播权重")
for ch in "医芯虎银月苹机诗":
    i = b.char2id.get(ch)
    if i is None:
        continue
    row = b.W[i]
    nz = np.nonzero(row)[0]
    top = nz[np.argsort(-row[nz])][:3]
    wn = b.Wn[i]
    print(f"  {ch}: 出边{len(nz):3d} 行和{row.sum():6.2f} "
          f"最大传播权重{wn.max():.5f} | "
          + ", ".join(f"{b.chars[j]}(W={row[j]:.2f},Ŵ={wn[j]:.5f})" for j in top))

print("\n# 要让一个新连接被想起来, 需要多大的 W?")
for ch in "医芯":
    i = b.char2id[ch]
    row_sum = float(b.W[i].sum())
    print(f"  {ch}: 行和={row_sum:.2f} → row^α={row_sum**f['alpha']:.2f}")
for ch in "芯苹":
    j = b.char2id[ch]
    col = float(b.W[:, j].sum())
    print(f"  {ch}: 列和={col:.2f} → col_eff^(1-α)={max(col, f['col_eff'][j])**(1-f['alpha']):.2f}")

# 实测: 单独激活种子后, 第 25 名的膜电位 (≈ 阈值)
for ch in "医芯":
    tr = b.think(ch, steps=20)
    print(f"  单独激活「{ch}」: 激活{tr.n_mean:.0f}字, 末态 θ={tr.theta_end:.4f}")

# 需要 W 多少才能达到 θ?
theta = b.think("医", steps=20).theta_end
i = b.char2id["医"]
j = b.char2id["芯"]
denom = (float(b.W[i].sum()) ** f["alpha"]) * (max(float(b.W[:, j].sum()), f["col_eff"][j]) ** (1 - f["alpha"]))
need_w = (theta * denom) ** (1.0 / f["gamma"])
print(f"\n  想用「医」把「芯」推过阈值 θ={theta:.4f}:")
print(f"    需要 Ŵ(医,芯) ≥ {theta:.4f}, 而分母 = {denom:.2f}")
print(f"    即需要 W(医,芯) ≥ {need_w:.3f}   (目前 0.055)")
print(f"    → 按 learn_eta=0.004/步, 每轮思考约加强 0.018, 约需 {need_w/0.018:.0f} 轮同类共现")
