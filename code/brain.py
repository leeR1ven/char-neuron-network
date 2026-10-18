# -*- coding: utf-8 -*-
"""字神经元网络 (char-neuron associative network)。

设计对应需求:
  1. 一个字 = 一个神经元; 语料里相邻出现的字之间建立有向连接 (i -> j)。
  2. "苹"后面总接"果" => W[苹][果] 强, W[果][苹] 弱; 方向由语料顺序决定。
  3. 权重增长机制: w = WMAX * c/(c+K) —— 出现越多越难涨 (饱和), 很少出现的连接
     头几次就拿到可观权重, 少见字因此不会被常见字压死。
  4. 特异性闸门: 用 PMI 抑制 "的->是" 这类到处都连的通用连接, 突出真正有指向的关系。
  5. 动力学: 传播激活 + 全局阈值 theta (整体调控, 不固定), 激活字太少就整体降阈值,
     太多就整体升阈值, 使激活规模稳定在一个区间。
  6. 抑制神经元: 每个字配一个抑制神经元; 语料里极少共现的字之间先验地建立
     "i -> 抑制神经元(k)" 的抑制连接, 并且在思考中继续学习。
  7. 实时学习: think() 过程中被先后激活的连接按同样"越涨越难涨"的规则加强,
     于是本来很少一起出现的字可以长出新的连接 (新思想)。
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict

import numpy as np

# ---------------------------------------------------------------- 参数

@dataclass
class Params:
    # --- 语料 / 词表 ---
    min_freq: int = 8            # 一个字至少出现多少次才算一个神经元
    max_vocab: int = 8000
    # --- 权重生长 (越涨越难涨) ---
    growth: str = "hill"         # "hill"(无上限凹增长+长尾回落, 推荐) | "bell" | "log" | "sat" | "peak"
    k_sat: float = 1.0           # 低端生长速度: 出现几次就有可观权重
    c_ref: float = 10000.0       # log 模式: 到这个量级封顶
    rise_k: float = 30.0         # hill: 上升段的半程位置 (到这里达到半高)
    rise_a: float = 0.5          # hill: 上升段陡度 (越小越平缓)
    cut_c: float = 3000000.0      # hill: 长尾回落的拐点 (共现超过这个量级开始明显变弱)
    cut_b: float = 1.0           # hill: 回落陡度
    bell_peak: float = 40.0      # bell 模式: 峰值位置
    bell_sigma: float = 2.5      # bell 模式: 峰宽
    peak_cut: float = 200000.0   # peak 模式: 共现超过这个量级开始明显回落
    peak_beta: float = 1.0       # peak 模式: 回落陡度
    peak_norm: float = 0.9       # peak 模式: 归一化, 让峰值 ≈ 1
    w_max: float = 1.0           # 单条连接权重上限
    c_min: int = 3               # 进图的最少共现次数
    top_k: int = 32              # 每个字最多保留多少条出边
    use_pmi: bool = True
    pmi_mid: float = 0.0         # PMI 中点
    pmi_temp: float = 1.2        # PMI 闸门陡度
    pmi_floor: float = 0.12      # 闸门下限 (不把连接完全掐死)
    pmi_pow: float = 1.4         # 闸门指数, 越大越突出"专属搭配"
    # --- 传播权重 ---
    use_norm: bool = False       # False = 不做任何归一化, 直接用连接强度 W 传播 (推荐)
    sharp_pow: float = 1.5       # 归一化模式下的锐化: Ŝ = W^γ
    norm_alpha: float = 0.75     # 归一化模式: Ŵ = Ŝ/(row^α · col_eff^(1-α))
    col_floor_k: float = 1.0     # col_eff = max(col, col_floor_k · col_typ), 防止冷僻字被抬高
    conf_k: float = 0.0          # 来源可靠度 conf=c/(c+K): 0 = 关闭 (默认不加这个机制)
    conf_pow: float = 0.5
    prior_mode: str = "off"      # 生僻目标抑制: 默认关闭; "sat"/"tail" 可开
    prior_k: float = 8000.0
    prior_pow: float = 0.6
    prior_floor: float = 3000.0
    # --- 全局阈值自适应 ---
    theta_init: float = 0.05
    theta_min: float = 0.001
    theta_max: float = 200.0       # 阈值上限 (新权重机制下驱动量级大, 上限要放宽)
    theta_lr: float = 0.5          # 阈值跟目标激活数的收敛速度 (0~1)
    target_active: int = 25        # 希望保持的激活字数
    act_eps: float = 0.35          # 激活判定阈值
    calibrate: bool = True         # 开局按驱动分布标定初始阈值
    # --- 传播动力学 ---
    steps: int = 20
    decay: float = 0.45            # 每步激活衰减 (持续激活是正常现象)
    out_soft: float = 0.15         # 输出整流软度 (越小响应越硬)
    gain_control: float = 0.01     # 除法式增益控制: 激活越多整体越难推高 (γ)
    seed_gain: float = 1.0         # 外部输入(种子字)强度
    seed_steps: int = 0            # 外部输入持续几步 (0 = 整轮一直供能)
    drive_gain: float = 1.0
    drive_mode: str = "sum"        # "sum": 多个字一起推(共识)  "lse": 软最大(只看最强证据)
    lse_beta: float = 12.0         # 软最大陡度: 越大越接近"取最大值"
    self_exc: float = 0.0          # 自连接保留比例 (反刍/持续激活)
    # --- 抑制神经元 ---
    use_inhibition: bool = True
    inh_per_target: int = 24     # 每个字的抑制神经元接收多少条来源连接
    inh_max_out: int = 96        # 每个字最多抑制多少个别的字
    inh_max: float = 1.0
    inh_gain: float = 0.6        # 抑制总增益
    inh_freq_k: float = 400.0    # 来源字要多常见才配得上"压制别人"
    # --- 实时学习 (纯时序赫布: 先亮的->刚亮的, 同时亮的弱加强) ---
    learn_eta: float = 0.02      # 连接增强步长 (一次思考小步, 反复共现才长成通路)
    learn_pow: float = 1.0       # 学习增量的软饱和指数 (越大越快变难涨)
    learn_scale: float = 1.0     # 学习增量的软饱和尺度: Δw = η·a_i·a_j/(1+w/scale)^pow
    w_max: float = 1e6           # 数值安全上限 (不是"权重上限", 正常永远碰不到)
    seq_hold: int = 2            # 语序模式: 每个字连续注入几步 (1=只注入一步)
    learn_inh_eta: float = 0.01  # 抑制连接增强步长
    learn_thresh: float = 0.5    # 多激活才算"一起激活"


# ---------------------------------------------------------------- 语料编码

def build_vocab(texts, min_freq: int, max_vocab: int = 8000):
    """统计字频, 返回 (chars 列表, char->id 字典, 频次数组)。id 0 起顺序分配。"""
    freq: dict[str, int] = {}
    for t in texts:
        for ch in t:
            freq[ch] = freq.get(ch, 0) + 1
    items = [(c, n) for c, n in freq.items() if n >= min_freq]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    items = items[:max_vocab]
    chars = [c for c, _ in items]
    char2id = {c: i for i, c in enumerate(chars)}
    counts = np.array([n for _, n in items], dtype=np.int64)
    return chars, char2id, counts


def vocab_from_freq(freq: np.ndarray, min_freq: int, max_vocab: int = 8000,
                    keep_ascii: bool = False):
    """由码位字频表构建词表。

    keep_ascii=False 时剔除拉丁字母与阿拉伯数字 —— 它们会自己抱成一团
    (单个字母之间互相连接), 污染"思考内容"。
    """
    cand = np.nonzero(freq >= min_freq)[0]
    if len(cand) == 0:
        return [], np.zeros(0, dtype=np.int64)
    if not keep_ascii:
        cand = np.array([c for c in cand
                         if c > 0x2E7F or (0x3000 <= c <= 0x303F)], dtype=np.int64)
    order = cand[np.argsort(-freq[cand], kind="stable")]
    order = order[:max_vocab]
    chars = [chr(int(c)) for c in order]
    return chars, freq[order].astype(np.int64)


class Encoder:
    """把文本快速编码成 id 序列 (查表, 未知字 = -1)。"""

    def __init__(self, char2id: dict[str, int]):
        lut = np.full(0x10000, -1, dtype=np.int32)
        for ch, i in char2id.items():
            o = ord(ch)
            if o < 0x10000:
                lut[o] = i
        self.lut = lut
        self.V = len(char2id)

    def encode(self, text: str) -> np.ndarray:
        arr = np.frombuffer(text.encode("utf-32-le", "ignore"), dtype="<u4")
        arr = np.minimum(arr, 0xFFFF).astype(np.int32)
        return self.lut[arr]


def accumulate_bigrams(ids: np.ndarray, C: np.ndarray, chunk: int = 4_000_000) -> None:
    """有向相邻共现计数 C[i,j] += 语料中 i 后面紧跟 j 的次数 (原地累加, 省内存)。

    ids 中 -1 表示"分隔符/未知字", 跨分隔符的配对不计。
    """
    V = C.shape[0]
    n = len(ids)
    for s in range(0, max(n - 1, 0), chunk):
        seg = ids[s : s + chunk + 1]
        a, b = seg[:-1], seg[1:]
        m = (a >= 0) & (b >= 0)
        if not m.any():
            continue
        keys = a[m].astype(np.int64) * V + b[m]
        uk, uc = np.unique(keys, return_counts=True)
        C.flat[uk] += uc.astype(np.int32)
        del keys, uk, uc


def count_bigrams(ids: np.ndarray, V: int, chunk: int = 4_000_000) -> np.ndarray:
    C = np.zeros((V, V), dtype=np.int32)
    accumulate_bigrams(ids, C, chunk)
    return C


def char_freqs(ids: np.ndarray, freq: np.ndarray) -> None:
    """按 Unicode 码位统计字频 (freq 长度 65536)。"""
    m = ids >= 0
    if m.any():
        np.add.at(freq, ids[m], 1)


# ---------------------------------------------------------------- 权重构建

def saturate(c: np.ndarray, k: float) -> np.ndarray:
    """饱和增长: c/(c+k)。出现越多, 边际增长越小; 少量出现也拿到可观权重。"""
    return c / (c + k)


def growth_curve(c: np.ndarray, p: Params) -> np.ndarray:
    """共现次数 -> 连接强度 (0..1)。可选的几种分布形式:

    log  : w = log(1+c/K) / log(1+Cref/K)      先快涨后慢涨, 到 Cref 封顶 (单调)
    sat  : w = c/(c+K)                          单调饱和
    peak : w = (1-exp(-c/K1)) * exp(-(c/C2)^β)  先快涨、到峰后**回落** (类似正态/对数正态)
           —— 很少见的连接也能拿到可观权重, 而"到处都出现"的搭配反而被压下去
    """
    c = np.maximum(c, 0.0)
    if p.growth == "sat":
        return saturate(c, p.k_sat).astype(np.float32)
    if p.growth == "hill":
        # 无上限的凹形增长 + 长尾回落 (不裁剪, 不吃到 1.0):
        #   上升段 w ∝ (c/K)^α / (1+(c/K)^α)  —— 边际增益递减, 出现几次就有可观权重
        #   回落段 × 1/(1+(c/C2)^β)           —— 到处都出现的搭配 (几十万次以上) 自动变弱
        rise = np.power(c / max(p.rise_k, 1e-6), p.rise_a)
        rise = rise / (1.0 + rise)
        fall = 1.0 / (1.0 + np.power(c / max(p.cut_c, 1e-6), p.cut_b))
        return (rise * fall).astype(np.float32)
    if p.growth == "bell":
        # 钟形 (对数正态): 出现 1~3 次就快速长起来, 几十~几百次最强,
        # 上万次开始回落, 几十万次以上(到处都出现的搭配)几乎归零。
        #   —— 不需要任何归一化, 曲线自己就把通用搭配压下去了。
        lc = np.log(np.maximum(c, 1e-6))
        mu = math.log(max(p.bell_peak, 1e-6))
        w = np.exp(-np.power(lc - mu, 2) / (2.0 * max(p.bell_sigma, 1e-6) ** 2))
        return np.clip(w, 0.0, 1.0).astype(np.float32)
    if p.growth == "peak":
        # 先按 log 曲线拿到"正常区间"的强弱, 再乘一个"过度常见就回落"的因子:
        #   出现几十~几万次: 基本不罚 (保持区分度)
        #   出现几十万次以上(的→是 这种到处都有的搭配): 迅速压下去
        base = np.minimum(np.log1p(c / max(p.k_sat, 1e-6)) /
                          math.log1p(max(p.c_ref, 1.0) / max(p.k_sat, 1e-6)), 1.0)
        fall = np.exp(-np.power(c / max(p.peak_cut, 1e-6), p.peak_beta))
        return np.clip(base * fall, 0.0, 1.0).astype(np.float32)
    num = np.log1p(c / max(p.k_sat, 1e-6))
    den = math.log1p(max(p.c_ref, 1.0) / max(p.k_sat, 1e-6))
    return np.minimum(num / max(den, 1e-9), 1.0).astype(np.float32)


def build_weights(C: np.ndarray, p: Params, char_freq: np.ndarray | None = None):
    """由共现计数构建权重矩阵。

    返回 (W, Wn, stats)
      W  : 原始权重 (可解释的"连接强度"), 稀疏化 (每行 top_k, 且 c >= c_min)
      Wn : 传播权重, 用于传播激活 —— 行列双侧归一化 + 来源可靠度 + 目标先验,
           常见字出边多被摊薄, 少见字出边集中且专属连接照样能推亮搭档。
    """
    V = C.shape[0]
    if V == 0:
        raise ValueError("空词表: 语料为空或 min_freq 过高")
    row = C.sum(axis=1).astype(np.float32)
    col = C.sum(axis=0).astype(np.float32)
    N = float(C.sum())

    Cf = C.astype(np.float32)
    W = growth_curve(Cf, p).astype(np.float32)     # 连接强度 (0..1)
    if p.use_pmi and N > 0:
        # 原地算 PMI 闸门 (float32, 避免 8000x8000 的 float64 临时数组)
        with np.errstate(divide="ignore", invalid="ignore"):
            den = np.outer(row, col)
            np.multiply(Cf, np.float32(N), out=Cf)
            np.divide(Cf, den, out=Cf)
            del den
            np.log2(Cf, out=Cf)
            Cf[~np.isfinite(Cf)] = -30.0
        np.subtract(Cf, np.float32(p.pmi_mid), out=Cf)
        np.divide(Cf, np.float32(max(p.pmi_temp, 1e-6)), out=Cf)
        np.negative(Cf, out=Cf)
        np.exp(Cf, out=Cf)
        np.add(Cf, np.float32(1.0), out=Cf)
        np.reciprocal(Cf, out=Cf)                  # gate = sigmoid(pmi)
        fl = np.float32(p.pmi_floor)
        np.multiply(Cf, np.float32(1.0) - fl, out=Cf)
        np.add(Cf, fl, out=Cf)
        np.power(Cf, np.float32(p.pmi_pow), out=Cf)
        np.multiply(W, Cf, out=W)
    del Cf
    W[C < p.c_min] = 0.0

    # 每行只留最强的 top_k 条出边
    if p.top_k < V:
        thr = np.partition(W, V - p.top_k, axis=1)[:, V - p.top_k][:, None]
        W[W < thr] = 0.0

    keep_ratio = float((W > 0).sum()) / (V * V)
    # 传播: 默认不做任何归一化, 直接用连接强度 W
    #   —— "该强该弱"完全由增长曲线决定: 少见搭配快速长大, 到处都出现的搭配自动回落。
    #   归一化只是可选的对照 (use_norm=True 时启用 行/列 双侧摊薄)。
    if not p.use_norm:
        Wn = W.copy()
        conf = np.ones(V, dtype=np.float32)
        prior = np.ones(V, dtype=np.float32)
        col_eff = np.ones(V, dtype=np.float32)
        stats = {
            "V": int(V), "bigrams_total": int(C.sum()),
            "edges_kept": int((W > 0).sum()), "keep_ratio": keep_ratio,
            "row_scale_median": 0.0, "col_typ": 0.0,
            "prop_max": float(Wn.max()),
            "prop_median": float(np.median(Wn[Wn > 0])) if (Wn > 0).any() else 0.0,
        }
        factors = {"conf": conf, "prior": prior, "col_eff": col_eff,
                   "alpha": 1.0, "gamma": 1.0, "use_norm": False}
        return W, Wn, stats, factors
    # 传播权重 (在"连接强度 W"之外再算一层"传播权重 Ŵ"):
    #   1) 锐化  Ŝ = W^γ             —— 强连接与弱连接拉开数量级
    #   2) 来源可靠度 conf=c/(c+K)    —— 只见了几次的新字少放火, 但它的专属连接仍能激活搭档
    #   3) 除以 row^α                —— 出边多的常见字被摊薄, 出边少的字更集中
    #   4) 除以 col_eff^(1-α)        —— 谁都想连的"万金油"被摊薄; 冷僻字不额外抬高
    S = np.power(W, np.float32(p.sharp_pow)).astype(np.float32)
    if p.conf_k > 0:
        conf = np.power(row / (row + np.float32(p.conf_k)), np.float32(p.conf_pow)).astype(np.float32)
        S *= conf[:, None]
    tf = char_freq.astype(np.float32) if char_freq is not None else (row + col).astype(np.float32)
    if p.prior_mode != "off" and p.prior_k > 0:
        # 目标侧先验: 压掉"只出现几十次、证据极弱"的字在思考里抢戏。
        #   sat : 全频段平滑压制 (常见字也被压一点)
        #   tail: 只压冷僻尾巴, 字频 >= prior_floor 完全不罚 (不碰正常字)
        if p.prior_mode == "tail":
            prior = np.minimum(np.float32(1.0),
                               np.power(tf / np.float32(max(p.prior_floor, 1.0)),
                                        np.float32(p.prior_pow))).astype(np.float32)
        else:
            prior = np.power(tf / (tf + np.float32(p.prior_k)), np.float32(p.prior_pow)).astype(np.float32)
        S *= prior[None, :]
    rs = S.sum(axis=1, keepdims=True)
    cs = S.sum(axis=0, keepdims=True)
    nz = cs[cs > 0]
    col_typ = float(np.percentile(nz, 50)) if nz.size else 1.0
    col_eff = np.maximum(cs, np.float32(p.col_floor_k * col_typ))
    a = float(p.norm_alpha)
    denom = np.power(np.maximum(rs, np.float32(1e-9)), np.float32(a)) * \
        np.power(col_eff, np.float32(1.0 - a))
    np.divide(S, denom, out=S)
    Wn = S

    stats = {
        "V": int(V),
        "bigrams_total": int(C.sum()),
        "edges_kept": int((W > 0).sum()),
        "keep_ratio": keep_ratio,
        "row_scale_median": float(np.median(rs[rs[:, 0] > 0])) if (rs > 0).any() else 0.0,
        "col_typ": col_typ,
        "prop_max": float(Wn.max()),
        "prop_median": float(np.median(Wn[Wn > 0])) if (Wn > 0).any() else 0.0,
    }
    # 保存换算因子, 供实时学习后精确重算 Ŵ (Ŵ = W^γ · conf · prior / (row^α · col_eff^(1-α)))
    factors = {
        "conf": conf if p.conf_k > 0 else np.ones(V, dtype=np.float32),
        "prior": prior if (p.prior_mode != "off" and p.prior_k > 0) else np.ones(V, dtype=np.float32),
        "col_eff": col_eff.astype(np.float32),
        "alpha": a,
        "gamma": float(p.sharp_pow),
    }
    return W, Wn, stats, factors


def build_inhibition(C: np.ndarray, char_freq: np.ndarray, p: Params):
    """抑制连接: i -> 抑制神经元(k)。

    打分 score_ik = 常见度(i) * (1 - 共现饱和) —— 即"i 很常见, 但和 k 几乎不搭",
    这种 i 会去压制 k。每个 k 取最强的 inh_per_target 个来源。
    返回 CSR: indptr, indices, data (源字 -> 目标字的抑制强度)。
    """
    V = C.shape[0]
    if not p.use_inhibition or V == 0:
        return np.zeros(V + 1, dtype=np.int64), np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)

    common = 1.0 - np.exp(-char_freq.astype(np.float64) / p.inh_freq_k)  # 常见度 0..1
    co = saturate(C.astype(np.float32), p.k_sat) * 0.5 + saturate(C.T.astype(np.float32), p.k_sat) * 0.5
    score = common[:, None] * (1.0 - co) * common[None, :]
    np.fill_diagonal(score, 0.0)

    m = min(p.inh_per_target, V)
    idx = np.argpartition(-score, m - 1, axis=0)[:m]           # 每个目标 k 的 top 来源
    tgt = np.repeat(np.arange(V), m)
    src = idx.T.reshape(-1)
    val = (score[src, tgt] * p.inh_max).astype(np.float32)

    # 每个源字再按强度截断出度, 控制计算量
    order = np.argsort(-val)
    src, tgt, val = src[order], tgt[order], val[order]
    seen = np.zeros(V, dtype=np.int64)
    keep = np.zeros(len(val), dtype=bool)
    for n in range(len(val)):
        s = src[n]
        if seen[s] < p.inh_max_out:
            keep[n] = True
            seen[s] += 1
    src, tgt, val = src[keep], tgt[keep], val[keep]

    order = np.argsort(src, kind="stable")
    src, tgt, val = src[order], tgt[order], val[order]
    indptr = np.zeros(V + 1, dtype=np.int64)
    np.add.at(indptr, src + 1, 1)
    indptr = np.cumsum(indptr)
    return indptr, tgt.astype(np.int32), val.astype(np.float32)


# ---------------------------------------------------------------- 大脑

@dataclass
class Step:
    t: int
    theta: float
    n_active: int
    top: list[tuple[str, float]]


@dataclass
class Trace:
    seeds: list[str]
    steps: list[Step] = field(default_factory=list)
    final: list[tuple[str, float]] = field(default_factory=list)
    order: list[tuple[str, int, float]] = field(default_factory=list)   # (字, 首次激活步, 强度)
    theta_end: float = 0.0
    n_active_hist: list[int] = field(default_factory=list)
    learned_edges: list[tuple[str, str, float, float]] = field(default_factory=list)

    @property
    def n_mean(self) -> float:
        tail = self.n_active_hist[len(self.n_active_hist) // 3:]
        return float(np.mean(tail)) if tail else 0.0

    @property
    def n_swing(self) -> float:
        tail = self.n_active_hist[len(self.n_active_hist) // 3:]
        return float(np.max(tail) - np.min(tail)) if tail else 0.0

    @property
    def n_first(self) -> int:
        return self.n_active_hist[0] if self.n_active_hist else 0

    def to_json(self):
        return {
            "seeds": self.seeds,
            "theta_end": self.theta_end,
            "n_active_first": self.n_first,
            "n_active_mean": round(self.n_mean, 1),
            "n_active_swing": round(self.n_swing, 1),
            "n_active_hist": self.n_active_hist,
            "steps": [
                {"t": s.t, "theta": round(s.theta, 5), "n_active": s.n_active,
                 "top": [[c, round(v, 3)] for c, v in s.top]}
                for s in self.steps
            ],
            "final": [[c, round(v, 3)] for c, v in self.final],
            "order": [[c, t, round(v, 3)] for c, t, v in self.order],
            "learned_edges": [
                [i, j, round(float(a), 4), round(float(b), 4)] for i, j, a, b in self.learned_edges
            ],
        }


class CharBrain:
    def __init__(self, chars: list[str], W: np.ndarray, Wn: np.ndarray,
                 inh=None, p: Params | None = None, meta: dict | None = None,
                 factors: dict | None = None):
        self.chars = chars
        self.char2id = {c: i for i, c in enumerate(chars)}
        self.W = W                  # 连接强度: 从语料学到的"知识", 实时学习改的就是它
        self.Wn = Wn                # 传播权重: 思考时 W 按扇入扇出/可靠度/先验换算后的有效传导率
        self.p = p or Params()
        self.V = len(chars)
        self.meta = meta or {}
        self.factors = factors
        if inh is None:
            self.inh_indptr = np.zeros(self.V + 1, dtype=np.int64)
            self.inh_idx = np.zeros(0, dtype=np.int32)
            self.inh_val = np.zeros(0, dtype=np.float32)
        else:
            self.inh_indptr, self.inh_idx, self.inh_val = inh
        # 抑制连接的来源下标 (向量化用)
        if len(self.inh_idx):
            self.inh_src = np.repeat(np.arange(self.V, dtype=np.int64),
                                     np.diff(self.inh_indptr))
        else:
            self.inh_src = np.zeros(0, dtype=np.int64)
        self.W_raw0 = W.copy()      # 训练前的权重快照, 用于对比"新思想"
        # 行和缓存 (实时学习后重算归一化用)
        self._row_sum = W.sum(axis=1)

    def rebuild_propagation(self) -> None:
        """按当前连接强度 W 重算传播矩阵 (实时学习之后调用)。"""
        f = self.factors or {}
        if not f.get("use_norm", True):
            self.Wn = self.W.copy()
            self._row_sum = self.W.sum(axis=1)
            return
        S = np.power(self.W, np.float32(f.get("gamma", self.p.sharp_pow))).astype(np.float32)
        S *= f["conf"][None, :]
        S *= f["prior"][None, :]
        rs = S.sum(axis=1, keepdims=True)
        a = float(f.get("alpha", self.p.norm_alpha))
        denom = np.power(np.maximum(rs, np.float32(1e-9)), np.float32(a)) * \
            np.power(f["col_eff"], np.float32(1.0 - a))
        np.divide(S, denom, out=S)
        self.Wn = S
        self._row_sum = self.W.sum(axis=1)

    # -------------------------------------------------- 输入输出
    def seed_vector(self, seeds, weights=None) -> np.ndarray:
        a = np.zeros(self.V, dtype=np.float32)
        if isinstance(seeds, str):
            seeds = list(seeds)
        for n, ch in enumerate(seeds):
            i = self.char2id.get(ch)
            if i is None:
                continue
            w = 1.0 if weights is None else float(weights[n])
            a[i] = max(a[i], w)
        return a

    def ids_to_str(self, ids) -> str:
        return "".join(self.chars[i] for i in ids)

    # -------------------------------------------------- 思考
    def think(self, seeds, steps: int | None = None, p: Params | None = None,
              learn: bool = False, record_every: int = 1, ext_hold: bool = True,
              calibrate: bool | None = None, sequence: bool = False) -> Trace:
        """传播激活。全局阈值 theta 每步按激活规模自适应调整 (整体调控, 不固定)。

        一步的动力学:
            drive_j = Σ_i a_i * Ŵ[i,j] / (1 + γ·Σa)      ← 传播 + 增益控制(整体抑制)
            drive_j -= inh_gain · inh_j                   ← 抑制神经元
            a_j     = clip( (decay·a_j + drive_j + 外部输入 - θ) / out_soft , 0, 1 )
            θ      += θ_lr · (n_active - target)/target · θ   ← 激活太多抬阈值, 太少降阈值
        """
        p = p or self.p
        steps = steps or p.steps
        ext_full = self.seed_vector(seeds)
        ext = ext_full
        seed_ids = np.nonzero(ext_full > 0)[0]
        a = ext.copy() if not sequence else np.zeros(self.V, dtype=np.float32)
        tr = Trace(seeds=[self.chars[i] for i in seed_ids])
        learned: dict[tuple[int, int], float] = {}
        prev_active = np.zeros(self.V, dtype=bool)
        session_t = np.full(self.V, -1, dtype=np.int32)   # 每个字本轮首次亮起的步
        seq_pos = 0            # sequence 模式下读到第几个字

        # 初始阈值标定: 让开局大约有 target_active 个字能越过阈值
        theta = p.theta_init
        if (p.calibrate if calibrate is None else calibrate) and len(seed_ids):
            d0 = ext @ self.Wn
            if (d0 > 0).any():
                thr = float(np.quantile(d0[d0 > 0], max(0.0, 1.0 - p.target_active / max(self.V, 1))))
                theta = float(np.clip(thr * 0.6, p.theta_min, p.theta_max))

        hist = []
        first_t = np.full(self.V, -1, dtype=np.int32)
        k = int(min(p.target_active, self.V))
        Wn = self.Wn
        for t in range(steps):
            src = a
            # 语序模式: 第 t 步只注入第 t 个字, 前面的字靠激活残留继续存在
            if sequence:
                ext = np.zeros(self.V, dtype=np.float32)
                if seq_pos < len(seed_ids):
                    cur = seed_ids[seq_pos]
                    ext[cur] = ext_full[cur] if ext_full[cur] > 0 else 1.0
                    if p.seq_hold > 1:
                        for q in range(1, p.seq_hold):
                            if seq_pos - q >= 0:
                                prev = seed_ids[seq_pos - q]
                                ext[prev] = max(ext[prev], ext_full[prev] * 0.5)
                    seq_pos += 1
            if p.drive_mode == "lse":
                # 软最大: 主要看"最强的那条证据", 避免常见字靠"源多"堆起来
                act = np.nonzero(src > 1e-3)[0]
                if len(act):
                    x = Wn[act] * src[act][:, None]
                    mx = x.max(axis=0)
                    drive = mx + np.log(np.exp((x - mx) * p.lse_beta).sum(axis=0)) / p.lse_beta
                    del x
                else:
                    drive = np.zeros(self.V, dtype=np.float32)
            else:
                drive = src @ Wn
            drive = drive * p.drive_gain
            s = float(a.sum())
            if p.gain_control > 0:
                drive = drive / (1.0 + p.gain_control * s)
            if p.use_inhibition and len(self.inh_idx):
                drive = drive - p.inh_gain * self._inhibition(a)
            if p.seed_steps and t >= p.seed_steps:
                ext_term = np.zeros(self.V, dtype=np.float32)   # 种子只点一下, 之后靠网络自己维持
            else:
                ext_term = ext * p.seed_gain
            pot = p.decay * a + drive + ext_term          # 阈值之前的膜电位

            # 全局阈值整体调控: 直接跟到"让 target_active 个字刚好越过阈值"的位置,
            # 再用 theta_lr 平滑跟进 —— 激活太多就整体抬高, 太少就整体压低。
            theta_star = float(np.partition(pot, self.V - k)[self.V - k]) - p.act_eps * p.out_soft
            theta = float(np.clip((1.0 - p.theta_lr) * theta + p.theta_lr * theta_star,
                                  p.theta_min, p.theta_max))

            a = np.clip((pot - theta) / p.out_soft, 0.0, 1.0)
            if ext_hold:
                a[seed_ids] = np.maximum(a[seed_ids], ext[seed_ids])

            active = a > p.act_eps
            n_active = int(active.sum())
            hist.append(n_active)
            newly = active & (first_t < 0)
            first_t[newly] = t

            if t % record_every == 0 or t == steps - 1:
                top = self._top_active(a, 14)
                tr.steps.append(Step(t=t, theta=float(theta), n_active=n_active, top=top))

            if learn:
                session_t[active & (session_t < 0)] = t
                self._learn_step(a, session_t, learned)
            prev_active = active

        tr.theta_end = theta
        tr.final = self._top_active(a, 60)
        tr.n_active_hist = hist
        if learn:
            # 学习改的是"连接强度 W"; 想完一整轮后按 W 精确重算"传播权重 Ŵ"
            self.relearn_normalization()
        # 按"首次激活的先后顺序"给出思考序列 (同一时刻内按强度排序)
        act_ids = np.nonzero(first_t >= 0)[0]
        order = sorted(act_ids.tolist(), key=lambda i: (int(first_t[i]), -float(a[i])))
        tr.order = [(self.chars[i], int(first_t[i]), float(a[i])) for i in order]
        tr.learned_edges = [
            (self.chars[i], self.chars[j], before, self.W[i, j])
            for (i, j), before in learned.items()
        ]
        return tr

    def _inhibition(self, a: np.ndarray) -> np.ndarray:
        """抑制神经元汇总: inh[k] = sum_i a_i * v[i->k] (向量化, 只算活跃源字)。"""
        inh = np.zeros(self.V, dtype=np.float32)
        if not len(self.inh_idx):
            return inh
        act = a > 0.05
        if not act.any():
            return inh
        m = act[self.inh_src]
        if m.any():
            np.add.at(inh, self.inh_idx[m], a[self.inh_src[m]] * self.inh_val[m])
        return inh

    def _top_active(self, a: np.ndarray, k: int) -> list[tuple[str, float]]:
        k = min(k, self.V)
        idx = np.argpartition(-a, k - 1)[:k]
        idx = idx[np.argsort(-a[idx])]
        return [(self.chars[i], float(a[i])) for i in idx if a[i] > 0.02]

    # -------------------------------------------------- 实时学习 (纯时序赫布)
    def _learn_step(self, a, session_t, learned):
        """每一时刻: 此刻还亮着的字之间, 按"谁先亮"定向加强。

          * 先亮的 i -> 后亮的 j : 持续加强 (一起亮着多久就加强多久)
          * 同时亮起的 i <-> j   : 双向互相加强 (没有先后, 同样全额)
          * 增量无上限但软饱和: dw = eta*a_i*a_j/(1+w/scale)^pow

        这样两个概念只要在句子里同时在场, 双向连接都会自然长出来;
        反复遇见会继续加强, 但增速越来越慢 (要有个度)。
        """
        p = self.p
        live = np.nonzero(a > p.act_eps)[0]
        if len(live) < 2:
            return
        ft = session_t[live]
        order = np.argsort(ft, kind="stable")
        nodes = live[order]
        fts = ft[order]
        n = len(nodes)
        src_i = np.repeat(nodes, n)
        tgt_i = np.tile(nodes, n)
        f_src = np.repeat(fts, n)
        f_tgt = np.tile(fts, n)
        keep = src_i != tgt_i
        fwd = keep & (f_src < f_tgt)                       # 先 -> 后
        same = keep & (f_src == f_tgt)                     # 同刻 -> 双向

        def apply(mask, scale):
            if not mask.any():
                return
            s, t_ = src_i[mask], tgt_i[mask]
            w_now = self.W[s, t_]
            inc = scale * a[s] * a[t_] / np.power(
                1.0 + w_now / max(p.learn_scale, 1e-6), p.learn_pow)
            m = inc > 1e-6
            if not m.any():
                return
            for x, y, d in zip(s[m].tolist(), t_[m].tolist(), inc[m].tolist()):
                before = float(self.W[x, y])
                if (x, y) not in learned:
                    learned[(x, y)] = before
                self.W[x, y] = min(p.w_max, before + float(d))

        apply(fwd, p.learn_eta)     # 先亮的 -> 后亮的
        apply(same, p.learn_eta)    # 同时亮的 <-> 互相加强 (同样全额)

        # 抑制连接: 现在活跃的字 i, 对"没被激活"的字 k 的抑制略微加强
        if p.use_inhibition and len(self.inh_idx):
            quiet = np.nonzero(~(a > p.act_eps))[0]
            if len(quiet):
                q = quiet[:24] if len(quiet) > 24 else quiet
                idx, val, ip = self.inh_idx, self.inh_val, self.inh_indptr
                for i in live[:40]:
                    st, en = ip[i], ip[i + 1]
                    if en <= st:
                        continue
                    tgt = idx[st:en]
                    hit = np.isin(tgt, q)
                    if hit.any():
                        newv = val[st:en][hit] + p.learn_inh_eta * np.power(
                            np.maximum(1.0 - val[st:en][hit] / p.inh_max, 0.0), p.learn_pow)
                        val[st + np.nonzero(hit)[0]] = np.minimum(p.inh_max, newv)

    def relearn_normalization(self):
        """实时学习之后重算传播权重 (有换算因子就精确重算, 否则退化为行归一化)。"""
        if self.factors:
            self.rebuild_propagation()
            return
        rs = self.W.sum(axis=1, keepdims=True)
        self.Wn = self.W.copy()
        np.divide(self.Wn, rs, out=self.Wn, where=rs > 0)
        self._row_sum = self.W.sum(axis=1)

    # -------------------------------------------------- 存取
    def save(self, path: str):
        np.savez_compressed(
            path, chars=np.array(self.chars), W=self.W, Wn=self.Wn,
            inh_indptr=self.inh_indptr, inh_idx=self.inh_idx, inh_val=self.inh_val,
            W_raw0=self.W_raw0, meta=np.array([json.dumps(self.meta, ensure_ascii=False)]),
            params=np.array([json.dumps(asdict(self.p), ensure_ascii=False)]),
            freq=getattr(self, "freq", np.zeros(self.V, dtype=np.int64)),
            factors=np.array([json.dumps(
                {k: (np.asarray(v).ravel().tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in (self.factors or {}).items()}, ensure_ascii=False)]),
        )

    @staticmethod
    def load(path: str) -> "CharBrain":
        z = np.load(path, allow_pickle=False)
        chars = [str(c) for c in z["chars"]]
        meta = json.loads(str(z["meta"][0]))
        raw = json.loads(str(z["params"][0]))
        known = set(asdict(Params()).keys())
        p = Params(**{k: v for k, v in raw.items() if k in known})
        factors = None
        if "factors" in z.files:
            try:
                d = json.loads(str(z["factors"][0]))
                factors = {k: (np.asarray(v, dtype=np.float32) if isinstance(v, list) else v)
                           for k, v in d.items()}
            except Exception:  # noqa: BLE001
                factors = None
        b = CharBrain(chars, z["W"], z["Wn"],
                      (z["inh_indptr"], z["inh_idx"], z["inh_val"]), p, meta,
                      factors=factors)
        b.W_raw0 = z["W_raw0"]
        if "freq" in z.files:
            b.freq = z["freq"]
        return b


# ---------------------------------------------------------------- 从语料训练

def train_from_corpus(lines, p: Params, log=print) -> CharBrain:
    t0 = time.time()
    texts = [ln.strip() for ln in lines if ln.strip()]
    chars, char2id, freq = build_vocab(texts, p.min_freq)
    log(f"[vocab] {len(chars)} 个神经元 (字频>={p.min_freq}), 语料 {sum(len(t) for t in texts):,} 字, "
        f"{time.time()-t0:.1f}s")
    enc = Encoder(char2id)
    V = len(chars)
    C = np.zeros((V, V), dtype=np.int32)
    for t in texts:
        ids = enc.encode(t)
        if len(ids) > 1:
            C += count_bigrams(ids, V)
    log(f"[count] bigram 统计完成, 非零 {int((C>0).sum()):,}, 总计 {int(C.sum()):,}, "
        f"{time.time()-t0:.1f}s")
    W, Wn, st, factors = build_weights(C, p, char_freq=freq)
    log(f"[weight] 保留连接 {st['edges_kept']:,} ({st['keep_ratio']*100:.3f}% 稠密), {time.time()-t0:.1f}s")
    inh = build_inhibition(C, freq, p)
    log(f"[inhibit] 抑制连接 {len(inh[1]):,}, {time.time()-t0:.1f}s")
    meta = {
        "vocab": V,
        "corpus_chars": int(sum(len(t) for t in texts)),
        "docs": len(texts),
        "bigrams": int(C.sum()),
        "edges": st["edges_kept"],
        "inh_edges": int(len(inh[1])),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    b = CharBrain(chars, W, Wn, inh, p, meta, factors=factors)
    b.freq = freq
    return b
