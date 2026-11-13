"""评估四种部署方式。

按「卡怎么分、每段用哪条切分」区分：

- ``uniform_kind``：全部卡、**同一套切分用于所有段**。这是纯 CP / 纯 EmbP。
  每段都用这条切分，对不吃这条切分的段就退化为「复制」。
- ``per_stage``：全部卡，但**每段各自选最合适的切分**。段间串行，
  因此卡可以在段与段之间被重新用途，这是标准 Megatron 式做法。
- ``stage_partition``：把段分给不同的卡组（空间划分）。单请求串行，
  每段只有部分卡参与。
- ``concurrent``：各组跑不同请求，真并行。
"""

from __future__ import annotations

from dataclasses import dataclass

from .cost import (
    Split,
    job_demand,
    stage_footprint,
    aggregate,
    footprint,
    helps,
    model_time,
    stage_time,
    stage_work,
)
from .machine import CardSet, Machine
from .models import Model, Shape

GB = 2**30


@dataclass(frozen=True)
class Evaluation:
    time_s: float
    peak_bytes: dict[int, float]
    plan: dict[str, Split]
    feasible: bool
    reason: str = ""
    link_load: float = 0.0
    rows: tuple[tuple[str, float, float, float], ...] = ()

    @property
    def peak_gb(self) -> float:
        return max(self.peak_bytes.values()) / GB if self.peak_bytes else 0.0


def _peak(model: Model, shape: Shape, plan: dict[str, Split], machine: Machine) -> tuple[bool, dict[int, float], str]:
    per_card = sum(footprint(model, shape, plan).values())
    peak = {c: per_card for c in range(machine.cards)}
    if per_card > machine.hbm_cap:
        return False, peak, f"每卡需 {per_card / GB:.1f} GB > {machine.hbm_cap / GB:.0f} GB"
    return True, peak, ""


def uniform_kind(model: Model, shape: Shape, kind: str, machine: Machine) -> Evaluation:
    """全部卡、所有段用同一条切分。"""
    if kind == "none":
        split = Split("none", 1)
        cards = 1
    else:
        split = Split(kind, machine.cards)
        cards = machine.cards
    total, rows = model_time(model, shape, split, machine)
    plan = {stage.name: split for stage in model.stages}
    feasible, peak, reason = _peak(model, shape, plan, machine)
    link = sum(row[3] for row in rows) / total if total > 0 else 0.0
    if cards < machine.cards:
        # 只用了 1 张卡：其余卡空闲，不算不可行
        pass
    return Evaluation(total, peak, plan, feasible, reason, link, rows)


def per_stage(model: Model, shape: Shape, machine: Machine) -> Evaluation:
    """全部卡，每段各选最合适的切分。"""
    plan: dict[str, Split] = {}
    for stage in model.stages:
        options = [Split("none", 1)]
        for kind in ("embp", "cp_q", "tp"):
            if helps(kind, stage.operator):
                options.append(Split(kind, machine.cards))
        scored: list[tuple[float, float, Split]] = []
        for option in options:
            memory = stage_footprint(stage, shape, option)
            # 单段就装不下的方案直接排除：查表切不切决定每卡是「全表」还是「表/N」，
            # 这是可行性问题，不是速度问题。
            if memory > machine.hbm_cap:
                continue
            work = aggregate(stage, stage_work(stage, shape), option, shape)
            t, _, _ = stage_time(work, option, machine, stage.operator == "lookup")
            scored.append((t, memory, option))
        if not scored:
            raise ValueError(f"{stage.name} 没有装得下的切分方案")
        fastest = min(item[0] for item in scored)
        # 时间在前 2% 以内的方案里，挑显存最小的。
        # 这条对查表段是决定性的：CP（表复制）与 EmbP（表切片）时间相同，
        # 但只有后者在大表时装得下。
        near = [item for item in scored if item[0] <= fastest * 1.02]
        plan[stage.name] = min(near, key=lambda item: (item[1], item[0]))[2]
    total, rows = model_time(model, shape, Split("none", 1), machine, per_stage=plan)
    feasible, peak, reason = _peak(model, shape, plan, machine)
    link = sum(row[3] for row in rows) / total if total > 0 else 0.0
    return Evaluation(total, peak, plan, feasible, reason, link, rows)


@dataclass(frozen=True)
class StagePlan:
    cards: CardSet
    split: Split


def stage_partition(
    model: Model,
    shape: Shape,
    assignment: dict[str, StagePlan],
    machine: Machine,
) -> Evaluation:
    """段级空间划分：每段在各自卡组上算，段间串行。"""
    used: set[int] = set()
    for stage in model.stages:
        if stage.name not in assignment:
            raise ValueError(f"阶段 {stage.name} 没有分配卡组")
        cards = set(range(assignment[stage.name].cards.start,
                          assignment[stage.name].cards.start + assignment[stage.name].cards.count))
        if used & cards:
            # 不同段允许共用卡组，但这里按「卡组划分」语义检查重叠仅用于记录
            pass
        used |= cards

    peak: dict[int, float] = {}
    total = 0.0
    comm_link = 0.0
    rows: list[tuple[str, float, float, float]] = []
    for stage in model.stages:
        item = assignment[stage.name]
        work = aggregate(stage, stage_work(stage, shape), item.split, shape)
        t, t_core, t_comm = stage_time(work, item.split, machine, stage.operator == "lookup")
        repeats = shape.layers if stage.name in model.per_layer else 1
        # 段间激活交接
        handoff = shape.n_tok * shape.d * shape.elem / machine.link_eff()
        t = t * repeats + handoff
        t_comm = t_comm * repeats + handoff
        rows.append((stage.name, t, t_core * repeats, t_comm))
        total += t
        comm_link += t_comm
        # 显存只算这一段落在它那组卡上的部分
        # stage_footprint 已经是「该组每卡」的量（embp 时已除以并行度），不再除卡数
        per_card = stage_footprint(stage, shape, item.split)
        for card in range(item.cards.start, item.cards.start + item.cards.count):
            peak[card] = peak.get(card, 0.0) + per_card
    total += model.n_kernel * machine.dispatch_s
    over = [c for c, b in peak.items() if b > machine.hbm_cap]
    feasible = not over
    reason = "" if feasible else f"卡 {over[0]} 需 {peak[over[0]] / GB:.1f} GB 超限"
    link = comm_link / total if total > 0 else 0.0
    plan = {name: item.split for name, item in assignment.items()}
    return Evaluation(total, peak, plan, feasible, reason, link, tuple(rows))


@dataclass(frozen=True)
class Job:
    model: Model
    shape: Shape
    cards: CardSet
    kind: str  # none / cp_q / embp / tp


def concurrent(jobs: tuple[Job, ...], machine: Machine) -> Evaluation:
    """多个请求各占一组卡，同时跑。makespan = 最慢那一组。

    链路争用：各组的通信时间占自身总时间的比例之和 > 1 时，按超出比例
    拉长（等效于共享同一条 HCCS 总线）。
    """
    used: set[int] = set()
    for job in jobs:
        cards = set(range(job.cards.start, job.cards.start + job.cards.count))
        if used & cards:
            raise ValueError(f"卡组重叠：{sorted(used & cards)}")
        used |= cards

    details: list[tuple[int, float, float, float]] = []
    link_sum = 0.0
    reasons: list[str] = []
    peak: dict[int, float] = {}
    for index, job in enumerate(jobs):
        split = Split(job.kind, job.cards.count) if job.kind != "none" else Split("none", 1)
        total, rows = model_time(job.model, job.shape, split, machine)
        plan = {stage.name: split for stage in job.model.stages}
        fp = sum(footprint(job.model, job.shape, plan).values())
        if fp > machine.hbm_cap:
            reasons.append(f"请求{index} 每卡 {fp / GB:.1f} GB 超限")
        link = sum(row[3] for row in rows) / total if total > 0 else 0.0
        link_sum += link
        details.append((index, total, link, fp))
        for card in range(job.cards.start, job.cards.start + job.cards.count):
            peak[card] = fp

    penalty = 1.0
    if link_sum > 1.0:
        penalty = link_sum
        details = [
            (i, t * (1.0 + (penalty - 1.0) * lk), lk, fp) for i, t, lk, fp in details
        ]
    times = {i: t for i, t, _, _ in details}
    makespan = max(times.values()) if times else 0.0
    card_time = {}
    for job, (_, t, _, _) in zip(jobs, details):
        for card in range(job.cards.start, job.cards.start + job.cards.count):
            card_time[card] = t
    plan_out = {stage.name: Split("none", 1) for stage in jobs[0].model.stages} if jobs else {}
    return Evaluation(
        makespan,
        peak,
        plan_out,
        not reasons,
        "；".join(reasons),
        min(1.0, link_sum),
        (),
    )


def best_uniform(model: Model, shape: Shape, machine: Machine) -> tuple[str, Evaluation]:
    """在可行方案里取最快的同构方案（纯 CP / 纯 EmbP / 纯 TP / 单卡）。"""
    best: tuple[str, Evaluation] | None = None
    for kind in ("none", "cp_q", "embp", "tp"):
        result = uniform_kind(model, shape, kind, machine)
        if not result.feasible:
            continue
        if best is None or result.time_s < best[1].time_s:
            best = (kind, result)
    if best is None:
        raise ValueError(f"{model.name} 在 {machine.cards} 卡上没有可行方案")
    return best

# ---------------------------------------------------------------- 共置（同一组卡跑多个请求）


@dataclass(frozen=True)
class CoJob:
    """要放进同一组卡的一个请求。`plan` 是它自己的逐段切分。"""

    name: str
    model: Model
    shape: Shape
    plan: dict[str, Split]


@dataclass(frozen=True)
class CoResult:
    makespan_s: float
    t_compute_s: float
    t_mem_s: float
    t_link_s: float
    profile: tuple[tuple[str, float, float, float], ...]  # (名字, t_c, t_m, t_l)
    feasible: bool
    reason: str = ""

    @property
    def bound(self) -> str:
        best = max(
            ("算力", self.t_compute_s),
            ("访存", self.t_mem_s),
            ("链路", self.t_link_s),
            key=lambda item: item[1],
        )
        return best[0]


def colocate(jobs: tuple[CoJob, ...], cards: int, machine: Machine) -> CoResult:
    """多个请求共享同一组卡。

    资源池模型：三种需求各自相加，再除以这组卡的容量，取最大值作为 makespan。
    这意味着「一个算力密集的请求 + 一个链路密集的请求」可以重叠推进，
    而「两个同类的」只能串行——这正是共置能否互补的判据。

    乐观之处：假设完美交织。因此它给出的是「互补收益的上界」，
    不是可实现值。用途是判断配对方向，不是预测绝对值。
    """
    if cards < 1:
        raise ValueError("卡数至少 1")
    flops = nbytes = link = 0.0
    profile: list[tuple[str, float, float, float]] = []
    reasons: list[str] = []
    cap = machine.hbm_cap
    for job in jobs:
        f, b, l = job_demand(job.model, job.shape, job.plan)
        flops += f
        nbytes += b
        link += l
        tc = f / (machine.p_eff() * cards)
        tm = b / (machine.bw_eff() * cards)
        tl = l / (machine.link_eff() * cards)
        profile.append((job.name, tc, tm, tl))
        fp = sum(footprint(job.model, job.shape, job.plan).values())
        if fp > cap:
            reasons.append(f"{job.name} 每卡 {fp / 2**30:.1f} GB 超限")

    t_compute = flops / (machine.p_eff() * cards) if flops > 0 else 0.0
    t_mem = nbytes / (machine.bw_eff() * cards) if nbytes > 0 else 0.0
    t_link = link / (machine.link_eff() * cards) if link > 0 else 0.0
    makespan = max(t_compute, t_mem, t_link)
    return CoResult(
        makespan,
        t_compute,
        t_mem,
        t_link,
        tuple(profile),
        not reasons,
        "；".join(reasons),
    )
