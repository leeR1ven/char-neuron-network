# -*- coding: utf-8 -*-
"""从缓存的共现矩阵快速重建模型 (不用重读语料), 用于比较权重侧的参数。

用法:
    python rebuild.py --counts models/counts_char_brain.npz --out models/char_brain.npz \
        --set prior_mode=off conf_k=0 sharp_pow=1.5
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import CharBrain, Params, build_inhibition, build_weights  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load_counts(path):
    z = np.load(path, allow_pickle=False)
    V = len(z["chars"])
    C = np.zeros((V, V), dtype=np.int32)
    C[z["rows"], z["cols"]] = z["data"]
    return C, [str(c) for c in z["chars"]], z["freq"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default=os.path.join(HERE, "models", "counts_char_brain.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "models", "char_brain.npz"))
    ap.add_argument("--set", nargs="*", default=[], help="覆盖参数, 形如 key=value")
    ap.add_argument("--meta", default=None, help="沿用哪个模型的 meta")
    args = ap.parse_args()

    C, chars, freq = load_counts(args.counts)
    p = Params()
    for kv in args.set:
        k, v = kv.split("=", 1)
        cur = getattr(p, k)
        if isinstance(cur, bool):
            v = v.lower() in ("1", "true", "yes")
        elif isinstance(cur, int) and not isinstance(cur, bool):
            v = int(float(v))
        elif isinstance(cur, float):
            v = float(v)
        setattr(p, k, v)
    print("参数覆盖:", {k: getattr(p, k) for k in
                     [kv.split("=")[0] for kv in args.set]} if args.set else "无")

    meta = {}
    if args.meta and os.path.exists(args.meta):
        meta = json.load(open(args.meta, encoding="utf-8"))
    W, Wn, st, factors = build_weights(C, p, char_freq=freq)
    inh = build_inhibition(C, freq, p)
    b = CharBrain(chars, W, Wn, inh, p, meta, factors=factors)
    b.freq = freq
    b.save(args.out)
    print(f"已保存 {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)  "
          f"连接 {st['edges_kept']:,}  抑制连接 {len(inh[1]):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
