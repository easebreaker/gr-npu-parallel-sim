"""卡级 Amdahl：只有部分段可并行时，冗余到底值不值得。

用户的设定：一台机器只跑同类模型，一次前向分 A–B–C–D 四段，只有 B 段能被
并行加速（比如注意力是唯一 L² 的段）。用 N 卡并行 B 之后，A、C、D 要在
其余 N-1 张卡上**重复执行**。

这个冗余有两个完全不同的解释，混淆它们会得出相反的结论：

1. **如果那些卡在等**（一次只有一个请求）：冗余就是纯粹浪费，
   上界由 Amdahl 给出 —— 加速比 ≤ (总时间)/(不可并行部分)。
2. **如果那些卡在算别的请求**（DP 批处理）：**不存在冗余**。
   每个副本处理**不同**的请求，总量随副本数线性增长。

文献里这一点是明说的：DP 的副本「处理不同请求，因此整体吞吐随副本增加，
但单请求延迟不变」。所以「如何优化冗余」的正确问法不是「怎么让 A/C/D 不重复」，
而是「什么时候该付这笔冗余」——即什么时候需要延迟，什么时候只需要吞吐。

本模块把这两个选择算成一对可比的数：
    latency_gain  = 用 CP 之后单请求快多少
    throughput_ratio = 用 DP 之后整机吞吐是 CP 的几倍
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .machine import Machine
from .models import Model, Shape
from .cost import Split, aggregate, stage_time, stage_work

GB = 2**30


@dataclass(frozen=True)
class StageCost:
    """一条请求在单卡、不切分时某一段的耗时。"""

    name: str
    seconds: float
    scalable: bool  # 这一段能否从 N 卡并行中获益

    def __post_init__(self) -> None:
        if self.seconds < 0:
            raise ValueError("时长不能为负")


@dataclass(frozen=True)
class Comparison:
    cards: int
    cp_latency: float  # 用 CP：一条请求占 N 卡
    dp_latency: float  # 用 DP：一条请求占 1 卡
    latency_gain: float  # dp_latency / cp_latency
    throughput_ratio: float  # DP 吞吐 / CP 吞吐
    amdahl_bound: float  # N→∞ 时的延迟增益上界
    non_scalable: float  # 不可并行部分（= Amdahl 里的 1−P）
    total: float

    def summary(self) -> str:
        return (
            f"N={self.cards}: CP {self.cp_latency * 1e3:8.3f} ms  "
            f"DP {self.dp_latency * 1e3:8.3f} ms  "
            f"延迟增益 {self.latency_gain:5.2f}×  "
            f"吞吐比 DP/CP {self.throughput_ratio:5.2f}×  "
            f"Amdahl 上界 {self.amdahl_bound:5.2f}×"
        )


def cp_latency(stages: tuple[StageCost, ...], cards: int, rho: float = 1.0) -> float:
    """一条请求用 N 卡并行可并行的段（CP），其余段在每张卡上重复。"""
    scale = cards ** rho
    total = 0.0
    for stage in stages:
        total += stage.seconds / scale if stage.scalable else stage.seconds
    return total


def dp_latency(stages: tuple[StageCost, ...]) -> float:
    """一条请求独占一张卡（DP）。没有任何冗余。"""
    return sum(stage.seconds for stage in stages)


def compare(
    stages: tuple[StageCost, ...],
    cards: int,
    rho: float = 1.0,
) -> Comparison:
    """CP 与 DP 的延迟/吞吐对照。

    `rho` 是可并行段的实际缩放指数：1.0 表示完美（线性），
    实测 GR 的注意力在 cp2→cp8 上约为 0.83（见 `实验数据-分析.md` 第六节）。
    """
    if cards < 1:
        raise ValueError("卡数至少 1")
    total = sum(stage.seconds for stage in stages)
    non_scalable = sum(stage.seconds for stage in stages if not stage.scalable)
    cp = cp_latency(stages, cards, rho)
    dp = dp_latency(stages)
    # CP：整机一次只服务 1 条请求 → 吞吐 1/cp
    # DP：整机同时服务 N 条请求，每条耗时 dp → 吞吐 N/dp
    throughput_ratio = cards * cp / dp if dp > 0 else 1.0
    return Comparison(
        cards=cards,
        cp_latency=cp,
        dp_latency=dp,
        latency_gain=dp / cp if cp > 0 else 1.0,
        throughput_ratio=throughput_ratio,
        amdahl_bound=dp / non_scalable if non_scalable > 0 else float("inf"),
        non_scalable=non_scalable,
        total=total,
    )


def stages_of(
    model: Model,
    shape: Shape,
    machine: Machine,
    scalable_ops: tuple[str, ...] = ("self_attn",),
) -> tuple[StageCost, ...]:
    """从模型结构取出单卡耗时，并标注哪些算子属于「可并行」。

    `scalable_ops` 决定哪一类算子算作 B 段。默认只有自注意力——
    这是用户设定的「只有 B 段被加速」。
    """
    out: list[StageCost] = []
    for stage in model.stages:
        split = Split("none", 1)
        work = aggregate(stage, stage_work(stage, shape), split, shape)
        t, _, _ = stage_time(work, split, machine, stage.operator == "lookup")
        repeats = shape.layers if stage.name in model.per_layer else 1
        seconds = t * repeats + machine.stage_fixed_s * repeats
        out.append(StageCost(stage.name, seconds, stage.operator in scalable_ops))
    out.append(StageCost("dispatch", model.n_kernel * machine.dispatch_s, False))
    return tuple(out)


def breakeven_cards(stages: tuple[StageCost, ...], rho: float = 1.0) -> float:
    """吞吐意义上，CP 开始「不亏」所需的卡数。

    解 N·cp(N) = dp(N)，即 N·(S + B/N^rho) = S + B，得
    N = ((S+B)/S)^(1/(1-rho))·... 当 rho=1 时方程无解——完美缩放下
    CP 的吞吐恒等于 DP 的 N 分之一再乘 N，即恰好相等，永不超过。

    因此返回的语义是：rho<1 时存在有限的 N 使 CP 吞吐追上 DP；
    rho=1 时返回 inf（永远追不上，只是打平）。
    """
    total = sum(stage.seconds for stage in stages)
    non_scalable = sum(stage.seconds for stage in stages if not stage.scalable)
    scalable = total - non_scalable
    if scalable <= 0 or non_scalable <= 0:
        return float("inf")
    if rho >= 1.0:
        return float("inf")
    # 吞吐比 = N·(S + B/N^rho)/(S+B)；求 =1 的根
    def ratio(n: float) -> float:
        return n * (non_scalable + scalable / n ** rho) / total

    lo, hi = 1.0, 1024.0
    if ratio(hi) >= 1.0:
        return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if ratio(mid) >= 1.0:
            hi = mid
        else:
            lo = mid
    return hi


def serves_latency(stages: tuple[StageCost, ...], cards: int, rho: float = 1.0) -> bool:
    """用 CP 是否在延迟上有实质收益（>10%）。"""
    result = compare(stages, cards, rho)
    return result.latency_gain > 1.10
