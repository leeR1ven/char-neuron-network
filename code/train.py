# -*- coding: utf-8 -*-
"""字神经元网络训练: 从 corpus/clean/*.txt 学习字与字之间的有向连接。

两遍流式处理 (内存友好, 可吃几十亿字):
  第1遍 统计字频 -> 词表 (一个字 = 一个神经元)
  第2遍 统计有向相邻共现 C[i,j] -> 饱和权重 + PMI 特异性闸门 + 抑制连接

每个来源按体积配额抽样, 保证语料多样 (新闻/百科/小说/诗词/论坛/论文都进得去)。

用法:
    python train.py --max-chars 1500000000 --min-freq 12 --max-vocab 8000
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brain import (Encoder, Params, accumulate_bigrams, build_inhibition, build_weights,
                   vocab_from_freq, CharBrain)  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus")
CLEAN = os.path.join(CORPUS, "clean")
MODELS = os.path.join(HERE, "models")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def plan_files(clean_dir: str, budget: int):
    """按文件体积比例分配字数配额。"""
    files = sorted(glob.glob(os.path.join(clean_dir, "*.txt")))
    sizes = []
    for p in files:
        try:
            sizes.append(os.path.getsize(p))
        except OSError:
            sizes.append(0)
    total = sum(sizes) or 1
    plan = []
    for p, s in zip(files, sizes):
        quota = int(budget * s / total)
        if quota > 20_000:
            plan.append((p, quota, s))
    return plan


def stream_ids(plan, encoder: Encoder | None, chunk_chars: int = 2_000_000,
               freq: np.ndarray | None = None, max_oov: float = 0.06):
    """按配额流式读取语料。encoder=None 时只做字频统计 (不编码成 id)。

    max_oov: 一行里"词表外字符"占比超过该值就丢掉 —— 用来剔除网页乱码/外文混排。
    """
    buf_ids: list[np.ndarray] = []
    buf_len = 0
    read_total = 0
    pending = 0
    dropped = 0
    for path, quota, _size in plan:
        got = 0
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                got += len(line)
                read_total += len(line)
                pending += len(line)
                if encoder is None:
                    arr = np.frombuffer(line.encode("utf-32-le", "ignore"), dtype="<u4")
                    arr = np.minimum(arr, 0xFFFF).astype(np.int32)
                    if freq is not None:
                        np.add.at(freq, arr, 1)
                else:
                    ids = encoder.encode(line)
                    if max_oov > 0 and len(ids) > 30:
                        ratio = float((ids < 0).sum()) / len(ids)
                        if ratio > max_oov:
                            dropped += 1
                            ids = None
                    if ids is not None:
                        buf_ids.append(ids)
                        buf_ids.append(np.array([-1], dtype=np.int32))
                        buf_len += len(ids) + 1
                if buf_len >= chunk_chars or pending >= chunk_chars:
                    if encoder is None:
                        yield None, read_total
                    else:
                        yield np.concatenate(buf_ids), read_total
                    buf_ids, buf_len, pending = [], 0, 0
                if got >= quota:
                    break
    print(f"[stream] 丢弃乱码行 {dropped:,}", flush=True)
    if encoder is None:
        if pending:
            yield None, read_total
    elif buf_ids:
        yield np.concatenate(buf_ids), read_total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chars", type=int, default=1_500_000_000)
    ap.add_argument("--min-freq", type=int, default=12)
    ap.add_argument("--max-vocab", type=int, default=8000)
    ap.add_argument("--k-sat", type=float, default=1.0)
    ap.add_argument("--c-ref", type=float, default=10000.0)
    ap.add_argument("--growth", default="log", choices=["log", "sat"])
    ap.add_argument("--sharp-pow", type=float, default=1.5)
    ap.add_argument("--norm-alpha", type=float, default=0.75)
    ap.add_argument("--conf-k", type=float, default=0.0)
    ap.add_argument("--prior-mode", default="off", choices=["off", "sat", "tail"])
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--c-min", type=int, default=3)
    ap.add_argument("--no-pmi", action="store_true")
    ap.add_argument("--save-counts", default=None, help="把共现矩阵存成稀疏 npz, 便于快速扫参")
    ap.add_argument("--keep-ascii", action="store_true",
                    help="保留拉丁字母/数字 (默认剔除, 它们会自己抱团污染思考内容)")
    ap.add_argument("--name", default="char_brain")
    args = ap.parse_args()

    os.makedirs(MODELS, exist_ok=True)
    plan = plan_files(CLEAN, args.max_chars)
    total_bytes = sum(s for _, _, s in plan)
    log(f"语料来源 {len(plan)} 个文件, 大小合计 {total_bytes/1e9:.2f} GB, "
        f"预算 {args.max_chars/1e9:.2f} G 字")

    t0 = time.time()
    freq = np.zeros(0x10000, dtype=np.int64)
    read = 0
    seen_log = 0
    for _ids, read in stream_ids(plan, None, chunk_chars=50_000_000, freq=freq):
        if read - seen_log >= 200_000_000:
            seen_log = read
            log(f"  字频统计: 已读 {read/1e9:.2f} G 字, 用时 {time.time()-t0:.0f}s")
    log(f"第1遍完成: 读入 {read:,} 字, 用时 {time.time()-t0:.0f}s")

    chars, counts = vocab_from_freq(freq, args.min_freq, args.max_vocab,
                                    keep_ascii=args.keep_ascii)
    V = len(chars)
    log(f"词表: {V} 个神经元 (字频>={args.min_freq}, 上限 {args.max_vocab}); "
        f"覆盖语料 {counts.sum()/max(read,1)*100:.2f}% 的字")
    char2id = {c: i for i, c in enumerate(chars)}
    enc = Encoder(char2id)

    C = np.zeros((V, V), dtype=np.int32)
    t1 = time.time()
    read2 = 0
    seen_log2 = 0
    for ids, read2 in stream_ids(plan, enc):
        accumulate_bigrams(ids, C)
        if read2 - seen_log2 >= 300_000_000:
            seen_log2 = read2
            log(f"  共现统计: 已读 {read2/1e9:.2f} G 字, 用时 {time.time()-t1:.0f}s")
    log(f"第2遍完成: bigram 非零 {int((C>0).sum()):,}, 总计 {int(C.sum()):,}, "
        f"用时 {time.time()-t1:.0f}s")

    p = Params(min_freq=args.min_freq, k_sat=args.k_sat, c_ref=args.c_ref,
               growth=args.growth, sharp_pow=args.sharp_pow, top_k=args.top_k,
               norm_alpha=args.norm_alpha, conf_k=args.conf_k,
               prior_mode=args.prior_mode, c_min=args.c_min, use_pmi=not args.no_pmi)
    if args.save_counts:
        r, c = np.nonzero(C)
        np.savez_compressed(args.save_counts, rows=r.astype(np.int32),
                            cols=c.astype(np.int32), data=C[r, c].astype(np.int32),
                            chars=np.array(chars), freq=counts)
        log(f"共现矩阵已缓存: {args.save_counts} (非零 {len(r):,})")
    W, Wn, st, factors = build_weights(C, p, char_freq=counts)
    log(f"权重: 保留连接 {st['edges_kept']:,} ({st['keep_ratio']*100:.3f}%), "
        f"用时 {time.time()-t1:.0f}s")
    inh = build_inhibition(C, counts, p)
    log(f"抑制连接: {len(inh[1]):,}")

    meta = {
        "semantics": "char-neuron associative network",
        "vocab": V,
        "corpus_chars_read": int(read),
        "corpus_chars_counted": int(read2),
        "corpus_files": len(plan),
        "corpus_bytes": int(total_bytes),
        "bigrams": int(C.sum()),
        "edges": int(st["edges_kept"]),
        "inh_edges": int(len(inh[1])),
        "k_sat": args.k_sat, "top_k": args.top_k, "c_min": args.c_min,
        "min_freq": args.min_freq, "use_pmi": not args.no_pmi,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - t0, 1),
    }
    brain = CharBrain(chars, W, Wn, inh, p, meta, factors=factors)
    brain.freq = counts
    out = os.path.join(MODELS, f"{args.name}.npz")
    brain.save(out)
    with open(os.path.join(MODELS, f"{args.name}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"已保存 {out} ({os.path.getsize(out)/1e6:.1f} MB)")

    # 速查: 几个常见字最强的后继
    for probe in ("苹", "果", "手", "机", "中", "国", "的"):
        i = char2id.get(probe)
        if i is None:
            continue
        row = W[i]
        idx = np.argsort(-row)[:6]
        top = [(chars[j], float(row[j])) for j in idx if row[j] > 0]
        log(f"  {probe} -> " + ", ".join(f"{c}({v:.2f})" for c, v in top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
