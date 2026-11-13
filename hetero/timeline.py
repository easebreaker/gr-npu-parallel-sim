"""时间线模型：每张卡什么时候忙、什么时候空，空出来的时间能不能塞下别的请求。

这一层与 `solver.py` 的区别：

- `solver` 假设「一个请求独占 N 卡」，算的是单请求的延迟。
- `timeline` 把卡看成独立资源，按区间分配，算的是**一段时间内多请求的 makespan
  与卡的利用率**。于是「某张卡提前空出来」这件事才有地方表达。

三个必须先分清的概念（它们常被混为一谈）：

1. **复制浪费**（replication waste）：一段不能被这条切分缩小，于是 N 张卡各做一遍。
   卡**没有空闲**，只是做了无用功。纯 CP 部署下的查表段就是这种。
2. **空闲**（idle）：某张卡在一段时间内没有被分配任何工作。
   只有「逐段不同卡数」或「段级分区」才会产生。
3. **气泡**（bubble）：流水线语义下，某张卡在等上游的激活。等价于「有工作但还轮不到」。

用户直觉里的「副卡先释放」对应的是 **2**，而它只在特定部署下出现。
本模块就是为了把这件事量化，而不是笼统地说「有空隙可以利用」。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .cost import Split, aggregate, stage_time, stage_work
from .machine import Machine
from .models import Model, Shape, Stage

GB = 2**30


@dataclass(frozen=True)
class StageJob:
    """一个待调度的阶段：需要 `degree` 张卡，耗时 `duration` 秒。"""

    name: str
    degree: int
    duration: float

    def __post_init__(self) -> None:
        if self.degree < 1:
            raise ValueError("卡数至少 1")
        if self.duration < 0:
            raise ValueError("时长不能为负")


@dataclass(frozen=True)
class Request:
    """一个请求的阶段序列（顺序依赖，段间必须串行）。"""

    name: str
    stages: tuple[StageJob, ...]

    @property
    def critical_path(self) -> float:
        return sum(job.duration for job in self.stages)


@dataclass(frozen=True)
class Interval:
    card: int
    start: float
    end: float
    label: str

    @property
    def length(self) -> float:
        return self.end - self.start


@dataclass
class Timeline:
    cards: int
    intervals: list[Interval] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    _by_card: list[list[Interval]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not self._by_card:
            self._by_card = [[] for _ in range(self.cards)]

    def add(self, interval: Interval) -> None:
        self.intervals.append(interval)
        self._by_card[interval.card].append(interval)

    def on_card(self, card: int) -> tuple[Interval, ...]:
        return tuple(self._by_card[card])

    def busy(self, card: int) -> float:
        return sum(iv.length for iv in self.intervals if iv.card == card)

    @property
    def makespan(self) -> float:
        return max((iv.end for iv in self.intervals), default=0.0)

    def busy_total(self) -> float:
        return sum(iv.length for iv in self.intervals)

    def utilization(self) -> float:
        span = self.makespan * self.cards
        return self.busy_total() / span if span > 0 else 0.0

    def idle_total(self) -> float:
        return self.makespan * self.cards - self.busy_total()

    def idle_gaps(self, card: int) -> tuple[tuple[float, float], ...]:
        """这张卡在 [0, makespan] 内的空闲区间。"""
        taken = sorted((iv.start, iv.end) for iv in self.on_card(card))
        gaps: list[tuple[float, float]] = []
        cursor = 0.0
        for start, end in taken:
            if start > cursor:
                gaps.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < self.makespan:
            gaps.append((cursor, self.makespan))
        return tuple(gaps)

    def largest_gap(self) -> float:
        return max((b - a for card in range(self.cards) for a, b in self.idle_gaps(card)), default=0.0)

    def per_card(self) -> tuple[tuple[int, float, float], ...]:
        return tuple(
            (card, self.busy(card), self.makespan - self.busy(card) if self.makespan else 0.0)
            for card in range(self.cards)
        )


def plan_durations(
    model: Model,
    shape: Shape,
    plan: dict[str, Split],
    machine: Machine,
) -> tuple[StageJob, ...]:
    """按逐段切分方案，算出每个阶段的卡数与时长。"""
    jobs: list[StageJob] = []
    for stage in model.stages:
        split = plan.get(stage.name, Split("none", 1))
        work = aggregate(stage, stage_work(stage, shape), split, shape)
        t, _, _ = stage_time(work, split, machine, stage.operator == "lookup")
        repeats = shape.layers if stage.name in model.per_layer else 1
        # 固定开销与卡数、L 都无关：它是 kernel 下发与同步的代价。
        fixed = machine.stage_fixed_s * repeats
        jobs.append(StageJob(stage.name, split.degree, t * repeats + fixed))
    jobs.append(StageJob("dispatch", 1, model.n_kernel * machine.dispatch_s))
    return tuple(jobs)


def as_request(name: str, jobs: tuple[StageJob, ...]) -> Request:
    return Request(name, jobs)


def greedy_schedule(
    requests: tuple[Request, ...],
    cards: int,
    machine: Machine | None = None,
) -> Timeline:
    """按到达顺序做表调度（list scheduling）。

    每个请求的段必须串行；每段需要 `degree` 张卡，取「这些卡都空下来」的最早时刻。
    段内不抢占。这是 vLLM / Orca 一类系统实际采用的策略的简化版
    （iteration-level 调度 + 顺序绑定），因此结论可以往真实系统上迁移。
    """
    timeline = Timeline(cards)
    free_at = [0.0] * cards
    for request in requests:
        ready = 0.0
        for job in request.stages:
            if job.degree > cards:
                timeline.rejected.append(f"{request.name}/{job.name}: 需要 {job.degree} 卡 > {cards}")
                ready = None  # type: ignore[assignment]
                break
            order = sorted(range(cards), key=lambda c: free_at[c])
            chosen = order[: job.degree]
            start = max(free_at[c] for c in chosen)
            start = max(start, ready)
            for card in chosen:
                timeline.add(Interval(card, start, start + job.duration, f"{request.name}/{job.name}"))
                free_at[card] = start + job.duration
            ready = start + job.duration
    return timeline


def static_schedule(
    durations: tuple[StageJob, ...],
    cards: int,
    requests: int,
) -> Timeline:
    """同构部署：多个相同请求依次独占全部卡。作为对照基线。"""
    timeline = Timeline(cards)
    cursor = 0.0
    for index in range(requests):
        for job in durations:
            if job.degree != cards:
                raise ValueError(f"段 {job.name} 的卡数是 {job.degree}，与 {cards} 不符")
            for card in range(cards):
                timeline.add(Interval(card, cursor, cursor + job.duration, f"req{index}/{job.name}"))
            cursor += job.duration
    return timeline

def _busy_on(timeline: Timeline, card: int, start: float, end: float) -> bool:
    return any(iv.start < end and start < iv.end for iv in timeline.on_card(card))


def _free_cards(timeline: Timeline, cards: int, start: float, end: float) -> list[int]:
    return [c for c in range(cards) if not _busy_on(timeline, c, start, end)]


def _feasible_at(
    timeline: Timeline,
    request: Request,
    start: float,
    cards: int,
) -> list[list[int]] | None:
    """在 `start` 时刻放这个请求，每段分到哪些卡。放不下则返回 None。"""
    assignment: list[list[int]] = []
    cursor = start
    for job in request.stages:
        if job.degree > cards:
            return None
        free = _free_cards(timeline, cards, cursor, cursor + job.duration)
        if len(free) < job.degree:
            return None
        assignment.append(free[: job.degree])
        cursor += job.duration
    return assignment


def backfill_schedule(
    requests: tuple[Request, ...],
    cards: int,
    machine: Machine | None = None,
) -> Timeline:
    """回填调度（backfill）：先放关键路径长的请求，再从最早可行时刻塞小的。

    与 `greedy_schedule` 的差别只在「什么时候允许后来的请求先开始」：
    - `greedy_schedule` 按到达顺序，后续请求要等前一个的所有段完成；
    - `backfill_schedule` 每个请求找**最早可行时刻**，可以塞进前面请求运行期间
      空出来的卡。

    这正是「副卡提前释放能否被利用」的判据：若所有段都要全部卡，
    backfill 与 greedy 结果相同（没有空隙可填）。

    候选时刻只取「0 与所有已占区间的结束时刻」——更细的扫描不会改变
    最早可行解，因为可行性只在区间端点处发生变化。
    """
    ordered = sorted(requests, key=lambda r: -r.critical_path)
    timeline = Timeline(cards)
    for request in ordered:
        candidates = sorted({0.0} | {iv.end for iv in timeline.intervals})
        placed = False
        for start in candidates:
            assignment = _feasible_at(timeline, request, start, cards)
            if assignment is None:
                continue
            cursor = start
            for job, chosen in zip(request.stages, assignment):
                for card in chosen:
                    timeline.add(
                        Interval(card, cursor, cursor + job.duration, f"{request.name}/{job.name}")
                    )
                cursor += job.duration
            placed = True
            break
        if not placed:
            timeline.rejected.append(f"{request.name}: 找不到可行时刻")
    return timeline

def stage_batched_schedule(
    requests: tuple[Request, ...],
    cards: int,
    machine: Machine | None = None,
) -> Timeline:
    """段级交错：把所有请求的同一阶段集中做完，再进下一段。

    与 `backfill_schedule` 的根本差别：后者要求一个请求的各段**连续**执行，
    前者允许不同请求处在不同阶段。

    只有当段的卡数需求不一致时才有意义——某段只用 1 卡时，
    集中做可以让 8 个请求共享那 1 卡的时间窗口，而其余 7 卡去跑别的段。

    若所有段都需要全部卡，本函数与"请求级串行"等价（没有可填的空）。
    """
    if not requests:
        return Timeline(cards)
    depth = len(requests[0].stages)
    for request in requests:
        if len(request.stages) != depth:
            raise ValueError("段级交错要求所有请求的段数相同（同一模型）")

    timeline = Timeline(cards)
    cursor = 0.0
    for index in range(depth):
        jobs = [request.stages[index] for request in requests]
        name = jobs[0].name
        for job in jobs:
            if job.degree != jobs[0].degree:
                raise ValueError(f"段 {name} 在不同请求上卡数不同，无法同批")
        degree = jobs[0].degree
        if degree > cards:
            timeline.rejected.append(f"{name}: 需要 {degree} 卡 > {cards}")
            continue
        per_round = cards // degree
        pending = list(range(len(requests)))
        while pending:
            wave = pending[:per_round]
            pending = pending[per_round:]
            # 同一波内各请求时长可以不同（长度重尾），波次时长取最长者；
            # 短的那个提前空出的卡在本实现里不复用（保守），
            # 这是「按长度分桶」要解决的问题。
            span = max(jobs[i].duration for i in wave)
            for slot, request_index in enumerate(wave):
                first = slot * degree
                for card in range(first, first + degree):
                    timeline.add(
                        Interval(
                            card,
                            cursor,
                            cursor + jobs[request_index].duration,
                            f"req{request_index}/{name}",
                        )
                    )
            cursor += span
    return timeline


def card_time_floor(requests: tuple[Request, ...], cards: int) -> float:
    """不可达到的理论下界：总卡时 / 卡数，且不小于最长请求的关键路径。

    它忽略段间先后依赖，因此偏乐观；用来衡量一个调度还有多少空间。
    """
    if not requests:
        return 0.0
    total = sum(
        job.duration * job.degree for request in requests for job in request.stages
    )
    longest = max(request.critical_path for request in requests)
    return max(total / cards, longest)
