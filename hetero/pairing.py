"""共置配对：多个请求共享同一组卡时，怎么配对才快。

用户提出的场景：四个模型，a/b 走 EmbP、c/d 走 CP，卡分两组、每组跑两个模型。
问的是「(a,b)+(c,d) 是不是不如 (a,c)+(b,d)」。

这类问题的收益有两个**互相独立**的来源，混在一起看会得出错误归因：

1. **负载均衡**：两组总工作量不等时，把便宜的挪到忙的那组能降 makespan。
   与「资源是否互补」无关，纯粹是配对让两组的量级接近。
2. **资源互补**：两条请求的瓶颈资源不同（一条算力、一条访存或链路）时，
   共享卡组可以重叠推进；同类瓶颈只能串行。

隔离方法：取两个**总时间相等**但瓶颈不同的请求。此时负载均衡效应消失，
剩下的差异只能来自互补。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from .cost import job_demand
from .machine import Machine
from .models import Model, Shape
from .solver import CoJob, CoResult, colocate, per_stage

GB = 2**30


@dataclass(frozen=True)
class Profile:
    """一个请求在一组卡上的资源画像。三项单位都是秒。"""

    name: str
    t_compute: float
    t_memory: float
    t_link: float

    @property
    def bound(self) -> str:
        name, _ = max(
            (("算力", self.t_compute), ("访存", self.t_memory), ("链路", self.t_link)),
            key=lambda item: item[1],
        )
        return name

    @property
    def span(self) -> float:
        """三项里的最大值，即这一请求自己的关键路径。"""
        return max(self.t_compute, self.t_memory, self.t_link)

    def vector(self) -> tuple[float, float, float]:
        return (self.t_compute, self.t_memory, self.t_link)

    def summary(self) -> str:
        return (
            f"算力 {self.t_compute * 1e3:7.2f} 访存 {self.t_memory * 1e3:7.2f} "
            f"链路 {self.t_link * 1e3:6.2f} [{self.bound}]"
        )


def profile(job: CoJob, cards: int, machine: Machine) -> Profile:
    flops, nbytes, link = job_demand(job.model, job.shape, job.plan)
    return Profile(
        job.name,
        flops / (machine.p_eff() * cards) if flops else 0.0,
        nbytes / (machine.bw_eff() * cards) if nbytes else 0.0,
        link / (machine.link_eff() * cards) if link else 0.0,
    )


def best_plan_job(name: str, model: Model, shape: Shape, machine: Machine) -> CoJob:
    """按逐段最优切分构造一个共置请求。"""
    return CoJob(name, model, shape, per_stage(model, shape, machine).plan)


def overlap_cosine(first: CoJob, second: CoJob, cards: int, machine: Machine) -> float:
    """两条请求资源向量的余弦相似度。1 = 完全同类，越小越互补。"""
    v1 = profile(first, cards, machine).vector()
    v2 = profile(second, cards, machine).vector()
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = sum(a * a for a in v1) ** 0.5
    n2 = sum(b * b for b in v2) ** 0.5
    if n1 == 0.0 or n2 == 0.0:
        return 1.0
    return dot / (n1 * n2)


@dataclass(frozen=True)
class Pairing:
    groups: tuple[tuple[str, ...], ...]
    makespan_s: float
    results: tuple[CoResult, ...]

    def describe(self) -> str:
        parts = []
        for names, result in zip(self.groups, self.results):
            parts.append(f"({'+'.join(names)}) {result.makespan_s * 1e3:7.2f} ms[{result.bound}]")
        return "  |  ".join(parts) + f"   → {self.makespan_s * 1e3:7.2f} ms"


def pairings_of_two(
    jobs: tuple[CoJob, ...],
    cards: int,
    machine: Machine,
) -> tuple[Pairing, ...]:
    """把 4 个请求分成两组、每组两个。共 3 种方式，按 makespan 升序返回。"""
    if len(jobs) != 4:
        raise ValueError("这里只处理 4 个请求分成两组")
    names = tuple(job.name for job in jobs)
    lookup = {job.name: job for job in jobs}
    first = names[0]
    out: list[Pairing] = []
    for partner in names[1:]:
        rest = tuple(n for n in names[1:] if n != partner)
        if len(rest) != 2:
            continue
        groups = ((first, partner), rest)
        results = tuple(
            colocate(tuple(lookup[n] for n in group), cards, machine) for group in groups
        )
        out.append(Pairing(groups, max(r.makespan_s for r in results), results))
    out.sort(key=lambda item: item.makespan_s)
    return tuple(out)


def best_pairing(jobs: tuple[CoJob, ...], cards: int, machine: Machine) -> Pairing:
    return pairings_of_two(jobs, cards, machine)[0]


def best_partition(
    jobs: tuple[CoJob, ...],
    cards: int,
    machine: Machine,
    group_sizes: tuple[int, ...],
) -> Pairing:
    """任意分组：按 `group_sizes` 穷举分配。每组卡数 = cards / 组数。"""
    if sum(group_sizes) != len(jobs):
        raise ValueError("组大小之和要等于请求数")
    names = tuple(job.name for job in jobs)
    lookup = {job.name: job for job in jobs}
    per_group = cards // len(group_sizes)
    best: Pairing | None = None
    for perm in itertools.permutations(names):
        groups: list[tuple[str, ...]] = []
        cursor = 0
        for size in group_sizes:
            groups.append(tuple(perm[cursor:cursor + size]))
            cursor += size
        results = tuple(
            colocate(tuple(lookup[n] for n in group), per_group, machine) for group in groups
        )
        candidate = Pairing(tuple(groups), max(r.makespan_s for r in results), results)
        if best is None or candidate.makespan_s < best.makespan_s:
            best = candidate
    assert best is not None
    return best


def decompose_speedup(
    alone_a: float,
    alone_b: float,
    together: float,
    profile_a: Profile,
    profile_b: Profile,
) -> dict[str, float]:
    """把「共置后变快」拆成均衡与互补两部分。

    - ``serial``：两条串行跑的总时间
    - ``balanced``：资源向量逐项相加后的最大分量——同类请求只能做到这个程度
    - ``together``：实际共置 makespan
    - ``gain_total`` = serial / together
    - ``gain_struct`` = serial / balanced（来自「不同的量落在不同资源上」）
    - ``gain_residual`` = balanced / together（同资源内的重叠，通常 ≈1）
    """
    serial = alone_a + alone_b
    v1 = profile_a.vector()
    v2 = profile_b.vector()
    balanced = max(a + b for a, b in zip(v1, v2))
    return {
        "serial": serial,
        "balanced": balanced,
        "together": together,
        "gain_total": serial / together if together else 1.0,
        "gain_struct": serial / balanced if balanced else 1.0,
        "gain_residual": balanced / together if together else 1.0,
    }

# ---------------------------------------------------------------- 对照实验


@dataclass(frozen=True)
class Ratio:
    """归一化画像：span 固定为 1，只描述三种资源的相对占比。"""

    name: str
    compute: float
    memory: float
    link: float

    def __post_init__(self) -> None:
        total = max(self.compute, self.memory, self.link)
        if total <= 0:
            raise ValueError("至少一项资源为正")
        # 归一化到 span = 1
        object.__setattr__(self, "compute", self.compute / total)
        object.__setattr__(self, "memory", self.memory / total)
        object.__setattr__(self, "link", self.link / total)

    def vector(self) -> tuple[float, float, float]:
        return (self.compute, self.memory, self.link)


def colocate_ratios(ratios: tuple[Ratio, ...]) -> dict[str, float]:
    """纯资源池：需求相加、取最大分量。返回各项与 makespan（单位 = span）。"""
    total_c = sum(r.compute for r in ratios)
    total_m = sum(r.memory for r in ratios)
    total_l = sum(r.link for r in ratios)
    makespan = max(total_c, total_m, total_l)
    return {
        "compute": total_c,
        "memory": total_m,
        "link": total_l,
        "makespan": makespan,
    }


def pairing_gain(same: tuple[Ratio, Ratio], cross: tuple[Ratio, Ratio]) -> dict[str, float]:
    """同质配对与异质配对的 makespan 比。两个对照下 span 之和相同，故均衡效应为零。"""
    a = colocate_ratios(same)
    b = colocate_ratios(cross)
    return {
        "same": a["makespan"],
        "cross": b["makespan"],
        "gain": a["makespan"] / b["makespan"] if b["makespan"] else 1.0,
    }
