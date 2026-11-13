"""单阶段的成本：roofline + 通信，且**按算子判定切分是否真的有效**。

这是整个仿真最关键的一处建模。一条切分对不同的算子作用完全不同：

| 切分 | 查表 | 线性投影 | 自注意力 | 其它段 |
|---|---|---|---|---|
| 不切（1 卡） | — | — | — | — |
| `embp`（切表） | **有效**：访存 /N | 无效：整段被复制 | 无效：被复制 | 被复制 |
| `cp_q`（切 Q） | 无效：每卡都查全表 | **有效**：token 维可切 | **有效**：算力 /N | 被复制 |
| `tp` | 无效 | **有效** | 无效 | 被复制 |

「被复制」的意思是：这 N 张卡各自把这一段完整做一遍，聚合工作量 ×N，
时间与单卡相同——**这就是纯 EmbP 部署在注意力段上的浪费，也是纯 CP 部署
在查表段上的浪费**。两者的浪费形态不同，因此存在逐段混搭的空间。

CP 还有一条不对称：只切 Q 时 KV 不切，每张卡都要读整段 KV，
因此**访存总量随 N 增长**。这条让 CP 在小 L 上亏。
"""

from __future__ import annotations

from dataclasses import dataclass

from .machine import Machine
from .models import Model, Shape, Stage

GB = 2**30
REPLICATE = "replicate"
EFFECTIVE = "effective"


@dataclass(frozen=True)
class Work:
    """聚合工作量：这一组 N 张卡合起来做了多少事。"""

    flops: float
    bytes: float
    comm: float
    note: str = ""


def _attn_flops(shape: Shape, q: float, kv: float) -> float:
    """两次 GEMM：qkᵀ 与 p·v，各 2 flops/乘加。"""
    return 4.0 * shape.batch * shape.heads * q * kv * shape.d_head


def _attn_bytes(shape: Shape, q: float, kv: float) -> float:
    """读 Q、K、V，写 O。"""
    per_head = shape.d_head * shape.elem
    return float(shape.batch * shape.heads * (q + 2.0 * kv + q) * per_head)


def stage_work(stage: Stage, shape: Shape) -> Work:
    """单卡、不切分时的「一次」工作量。"""
    op = stage.operator
    if op == "lookup":
        tokens = shape.tokens_of(stage.tokens)
        return Work(0.0, 2.0 * tokens * shape.d * shape.elem, 0.0, "MTE（随机访存）")

    if op == "linear":
        tokens = shape.tokens_of(stage.tokens)
        flops = 2.0 * tokens * shape.d * stage.out_width
        nbytes = (tokens * shape.d + shape.d * stage.out_width + tokens * stage.out_width) * shape.elem
        return Work(flops, nbytes, 0.0, "Cube")

    if op == "self_attn":
        seq = shape.seq_of(stage.seq)
        return Work(_attn_flops(shape, seq, seq), _attn_bytes(shape, seq, seq), 0.0, "Cube ↔ MTE")

    if op == "group_attn":
        tokens = shape.seq_of(stage.seq)
        groups = float(shape.groups)
        q = groups * tokens
        return Work(
            4.0 * shape.batch * shape.heads * groups * tokens * tokens * shape.d_head,
            _attn_bytes(shape, q, tokens),
            0.0,
            "Cube（∝ L·T）",
        )

    if op == "cross_attn":
        kv = shape.seq_of(stage.seq) * shape.batch
        q = float(shape.batch * shape.short_len)
        return Work(_attn_flops(shape, q, kv), _attn_bytes(shape, q, kv), 0.0, "Cube 或 MTE")

    if op == "permute":
        tokens = shape.tokens_of(stage.tokens)
        return Work(0.0, 2.0 * tokens * shape.d * shape.elem, 0.0, "MTE（纯搬运）")

    raise KeyError(op)


@dataclass(frozen=True)
class Split:
    kind: str  # none / cp_q / embp / tp
    degree: int = 1

    def __post_init__(self) -> None:
        if self.kind not in ("none", "cp_q", "embp", "tp"):
            raise ValueError(f"未知切分 {self.kind}")
        if self.degree < 1:
            raise ValueError("并行度至少 1")
        if self.kind != "none" and self.degree < 2:
            raise ValueError(f"{self.kind} 的并行度至少 2")

    def __str__(self) -> str:
        return "不切分" if self.kind == "none" else f"{self.kind}×{self.degree}"


def helps(split_kind: str, operator: str) -> bool:
    """这条切分对这个算子是否真的减少工作量。

    查表是**按 token 完全可切**的：每个 token 独立查表，与 token 维怎么分无关。
    因此 CP（切 token）对查表也有效——8 卡分担一个请求的 token 就是 8 倍快。
    这一点与「复制」不同：复制是每卡都做全量，才是浪费。

    EmbP 与 CP 在**时间**上等价（都按 N 摊），差别只在两处：
      - 显存：EmbP 让每卡只持表/N，CP 则每卡持全表
      - 通信：EmbP 需要一次 All-to-All 把查到的向量送回，CP 不需要

    所以查表段用 EmbP 还是 CP，是「显存 vs 通信」的取舍，不是时间取舍。
    """
    if split_kind == "embp":
        return operator == "lookup"
    if split_kind == "cp_q":
        return operator in ("self_attn", "group_attn", "cross_attn", "linear", "lookup")
    if split_kind == "tp":
        return operator == "linear"
    return False


def aggregate(stage: Stage, work: Work, split: Split, shape: Shape) -> Work:
    """把单卡工作量换成「这一组 N 张卡上的聚合工作量」。"""
    n = split.degree
    if n == 1:
        return work

    if not helps(split.kind, stage.operator):
        # 被复制：N 张卡各做一遍，聚合 ×N，时间不变
        return Work(work.flops * n, work.bytes * n, 0.0, work.note + "（复制）")

    if split.kind == "embp":
        # 表切开：聚合访存不变，另加 All-to-All
        moved = work.bytes * (n - 1.0) / n
        return Work(work.flops, work.bytes, moved, work.note)

    if split.kind == "tp":
        moved = 2.0 * work.bytes * (n - 1.0) / n
        return Work(work.flops, work.bytes, moved, work.note)

    if split.kind == "cp_q":
        if stage.operator in ("self_attn", "cross_attn", "group_attn"):
            # 算力按 N 摊（聚合不变）；KV 不切，聚合访存 ×(1+N)/2
            kv_factor = (1.0 + n) / 2.0
            moved = work.bytes * (n - 1.0) / n * 0.5  # KV 交换
            return Work(work.flops, work.bytes * kv_factor, moved, work.note + f"（KV×{kv_factor:.1f}）")
        # 线性类按 token 切：聚合工作量不变，无通信
        return Work(work.flops, work.bytes, 0.0, work.note)

    raise KeyError(split.kind)


def stage_time(
    work: Work,
    split: Split,
    machine: Machine,
    random_access: bool = False,
) -> tuple[float, float, float]:
    """返回 (总时间, 计算/访存时间, 通信时间)，单位秒。

    `random_access` 为真时按随机 gather 的等效带宽算——查表是随机访存，
    与流式搬运不能共用一个带宽。
    """
    n = float(split.degree)
    compute_share = machine.p_eff() * n
    mem_share = (machine.gather_bw() if random_access else machine.bw_eff()) * n
    t_flops = work.flops / compute_share if work.flops > 0.0 else 0.0
    t_bytes = work.bytes / mem_share if work.bytes > 0.0 else 0.0
    t_core = max(t_flops, t_bytes)
    # 单个 kernel 有最短时长：低于它之后加卡不再有效，只是空转。
    t_core = max(t_core, machine.min_stage_s)
    t_comm = work.comm / (machine.link_eff() * n) if work.comm > 0.0 else 0.0
    return t_core + t_comm, t_core, t_comm


def model_time(
    model: Model,
    shape: Shape,
    split: Split,
    machine: Machine,
    per_stage: dict[str, Split] | None = None,
) -> tuple[float, tuple[tuple[str, float, float, float], ...]]:
    """整条前向的时间。`per_stage` 非空时按阶段覆盖 `split`。"""
    rows: list[tuple[str, float, float, float]] = []
    total = 0.0
    for stage in model.stages:
        used = (per_stage or {}).get(stage.name, split)
        work = aggregate(stage, stage_work(stage, shape), used, shape)
        t, t_core, t_comm = stage_time(work, used, machine, stage.operator == "lookup")
        repeats = shape.layers if stage.name in model.per_layer else 1
        rows.append((stage.name, t * repeats, t_core * repeats, t_comm * repeats))
        total += t * repeats
    total += model.n_kernel * machine.dispatch_s
    return total, tuple(rows)


def stage_footprint(stage: Stage, shape: Shape, split: Split) -> float:
    """单张卡上，**只属于这一段**的常驻字节。"""
    repeats = shape.layers if False else 1
    if stage.operator == "lookup":
        share = split.degree if split.kind == "embp" else 1
        return stage.table_bytes / share
    total = stage.weight_bytes
    if stage.operator in ("self_attn", "group_attn", "cross_attn"):
        seq = shape.seq_of(stage.seq)
        total += 2.0 * shape.batch * seq * shape.heads * shape.d_head * shape.elem
    return total


def footprint(model: Model, shape: Shape, plan: dict[str, Split] | Split) -> dict[str, float]:
    """一张卡上的常驻字节。表是否切开由那段自己的切分决定。"""
    def split_of(name: str) -> Split:
        return plan.get(name, Split("none")) if isinstance(plan, dict) else plan

    table = 0.0
    weights = 0.0
    kv = 0.0
    for stage in model.stages:
        used = split_of(stage.name)
        repeats = shape.layers if stage.name in model.per_layer else 1
        if stage.operator == "lookup":
            share = used.degree if used.kind == "embp" else 1
            table += stage.table_bytes / share
        else:
            weights += stage.weight_bytes * repeats
        if stage.operator in ("self_attn", "group_attn", "cross_attn"):
            seq = shape.seq_of(stage.seq)
            kv += 2.0 * shape.batch * seq * shape.heads * shape.d_head * shape.elem * repeats
    return {"table": table, "weights": weights, "kv": kv}

def job_demand(
    model: Model,
    shape: Shape,
    plan: dict[str, Split],
) -> tuple[float, float, float]:
    """一个请求在一个卡组上的总需求：(flops, bytes, link_bytes)。

    `plan` 是该请求自己的逐段切分——不同模型的最优方案不同，
    但需求都用绝对量表示，便于共置时相加。
    """
    flops = nbytes = link = 0.0
    for stage in model.stages:
        split = plan.get(stage.name, Split("none", 1))
        work = aggregate(stage, stage_work(stage, shape), split, shape)
        repeats = shape.layers if stage.name in model.per_layer else 1
        flops += work.flops * repeats
        nbytes += work.bytes * repeats
        link += work.comm * repeats
    return flops, nbytes, link

def max_tp_degree(shape: Shape, per_head_splittable: bool = False) -> int:
    """TP 在推荐模型上的实际上限：受注意力头数限制。

    来源：NVIDIA recsys-examples issue #71 在实现 HSTU 的 TP 时明确指出
    「The TP size is limited by the attention heads」。原因是被切的是
    Q/K/V 的输出维，而每个头是最小不可分单位（除非进一步切 head_dim）。

    `per_head_splittable=True` 时放宽到 head × head_dim（即允许切头内维），
    但那会带来额外的归约通信，实际很少用。
    """
    if per_head_splittable:
        return max(1, shape.heads * shape.d_head)
    return max(1, shape.heads)
