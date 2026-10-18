# -*- coding: utf-8 -*-
"""一键流水线: 清洗语料 -> 训练 -> 实验 -> 生成报告。

用法:
    python pipeline.py --max-chars 1500000000 --name char_brain
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(title: str, args: list[str]) -> None:
    print(f"\n{'='*80}\n### {title}\n{'='*80}", flush=True)
    t0 = time.time()
    r = subprocess.run([PY] + args, cwd=HERE)
    print(f"--- {title} 结束 (exit={r.returncode}, {time.time()-t0:.0f}s)", flush=True)
    if r.returncode != 0:
        print(f"!!! {title} 失败", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chars", type=int, default=1_500_000_000)
    ap.add_argument("--min-freq", type=int, default=12)
    ap.add_argument("--max-vocab", type=int, default=8000)
    ap.add_argument("--name", default="char_brain")
    ap.add_argument("--skip-prepare", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    if not args.skip_prepare:
        run("1/4 清洗语料 (统一成 一篇一行)", ["tools/prepare_corpus.py", "--workers", "3"])
    if not args.skip_train:
        run("2/4 训练字神经元网络", [
            "train.py", "--max-chars", str(args.max_chars),
            "--min-freq", str(args.min_freq), "--max-vocab", str(args.max_vocab),
            "--name", args.name,
            "--save-counts", os.path.join("models", f"counts_{args.name}.npz")])
    model = os.path.join("models", f"{args.name}.npz")
    run("3/4 思考实验", ["run_experiments.py", "--model", model,
                         "--out", os.path.join("results", args.name)])
    run("4/4 生成报告", ["report.py", "--model", model,
                         "--results", os.path.join("results", args.name),
                         "--out", os.path.join("结果", f"{args.name}_实验报告.md")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
