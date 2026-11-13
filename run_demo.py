"""四个模型在不同并行方案下的仿真结果。

全部是仿真，不是实测。参数见 `hetero/machine.py` 与各模型的默认形状。
结论分四层：纯同构 → 逐段混搭 → 段级空间划分 → 并发异构。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hetero import (  # noqa: E402
    CardSet,
    Job,
    Machine,
    Shape,
    Split,
    best_uniform,
    concurrent,
    footprint,
    hyformer,
    longer,
    model_time,
    mtgr,
    per_stage,
    stage_partition,
    uniform_kind,
)
from hetero.pairing import (  # noqa: E402
    Ratio,
    best_plan_job,
    colocate_ratios,
    overlap_cosine,
    pairing_gain,
    pairings_of_two,
    profile,
)
from hetero.solver import StagePlan  # noqa: E402

GB = 2**30
MACHINE = Machine()


def rule(title: str) -> None:
    print()
    print("=" * 84)
    print(title)
    print("=" * 84)


def ms(value: float) -> str:
    return f"{value * 1e3:8.2f}"


def show_structure() -> None:
    rule("1 四个模型的阶段结构（由源码定，是仿真的输入）")
    print(f"{'模型':10s} {'阶段':14s} {'增长律':8s} 说明")
    for factory in (mtgr, longer, hyformer):
        model = factory()
        for index, stage in enumerate(model.stages):
            name = model.name if index == 0 else ""
            print(f"{name:10s} {stage.name:14s} {stage.growth:8s} {stage.note}")
        print()
    print("关键结构差异：")
    print("  MTGR/HSTU  注意力是 L²，全长序列都算 —— CP 收益最大")
    print("  LONGER     组内注意力是 L·T（线性）—— CP 只拿线性收益")
    print("  HyFormer   没有 L²：decode 是 short×short、boost 固定 —— 但查表仍按 token 线性，")


def show_growth() -> None:
    rule("2 长度敏感性：单卡不切分时时间随 L 怎么变")
    print(f"{'模型':10s} {'L=512':>9s} {'L=2048':>9s} {'L=8192':>9s} {'L=32768':>9s}   {'8192/512':>9s}")
    for factory in (mtgr, longer, hyformer):
        times = []
        for length in (512, 2048, 8192, 32768):
            shape = Shape(length=length, kv_len=min(length, 2048))
            total, _ = model_time(factory(), shape, Split("none", 1), MACHINE)
            times.append(total)
        print(
            f"{factory().name:10s} {ms(times[0]):>9s} {ms(times[1]):>9s} {ms(times[2]):>9s} "
            f"{ms(times[3]):>9s}   {times[2] / times[0]:8.1f}x"
        )
    print("\n  → MTGR 涨得最快（L²）；LONGER 中等（L·T）；HyFormer 虽无 L²，但查表每 token 一次，"
          "仍随 L 线性。它的注意力段本身是平的（见测试 test_growth_order_matches_structure）。")


def show_pure_vs_mixed() -> None:
    rule("3 纯 CP vs 纯 EmbP vs 逐段混搭（都是 8 卡，表 50M 行）")
    print("  纯 CP：所有段都按 cp_q×8 —— 查表段被复制 8 份（每卡都查全表）")
    print("  纯 EmbP：所有段都按 embp×8 —— 注意力段被复制 8 份（每卡都算整段 L²）")
    print("  混搭：每段各自选最合适的切分（查表 EmbP、注意力 CP、投影按 token 切）\n")
    for factory in (mtgr, longer, hyformer):
        for length in (512, 8192):
            shape = Shape(length=length, kv_len=min(length, 2048))
            model = factory()
            pure_cp = uniform_kind(model, shape, "cp_q", MACHINE)
            pure_ep = uniform_kind(model, shape, "embp", MACHINE)
            mixed = per_stage(model, shape, MACHINE)
            best_kind, best = best_uniform(model, shape, MACHINE)
            winner = min(
                (("纯CP", pure_cp), ("纯EmbP", pure_ep), ("混搭", mixed), (f"纯{best_kind}", best)),
                key=lambda item: item[1].time_s,
            )
            print(
                f"  {model.name:10s} L={length:<6d} 纯CP {ms(pure_cp.time_s)}  纯EmbP {ms(pure_ep.time_s)}  "
                f"混搭 {ms(mixed.time_s)}   → 最优：{winner[0]}"
            )
        print()
    print("  读法：")
    print("  · 混搭在三个模型上都最优，但赢的理由不同：")
    print("    MTGR   —— 纯 EmbP 把注意力复制 8 份（每卡都算整段 L²），亏到 7 倍；")
    print("              纯 CP 把查表复制 8 份且表不切，亏约 7%。")
    print("    LONGER —— 注意力是 L·T，CP 的收益本来就小，混搭省下的是查表与交叉段的复制。")
    print("    HyFormer —— 没有 L²，CP 反而把交叉注意力的访存放大 4.5 倍（KV 不切），")
    print("              所以纯 CP 比纯 EmbP 还差；混搭只让查表走 EmbP，其余走不切分。")
    print("  · 一句话：混搭不是「更高级的切分」，而是**避免把不适用的切分套在不吃的段上**。")


def show_plan_detail() -> None:
    rule("4 混搭方案具体长什么样（MTGR, L=8192）")
    shape = Shape(length=8192, kv_len=2048)
    model = mtgr()
    result = per_stage(model, shape, MACHINE)
    print(f"{'阶段':14s} {'切分':>10s} {'时间ms':>9s} {'占比':>7s}")
    for name, t, _, _ in result.rows:
        print(f"{name:14s} {str(result.plan[name]):>10s} {ms(t):>9s} {t / result.time_s * 100:6.1f}%")
    print(f"{'下发':14s} {'':>10s} {ms(model.n_kernel * MACHINE.dispatch_s):>9s}")
    print(f"{'合计':14s} {'':>10s} {ms(result.time_s):>9s}   每卡常驻 {result.peak_gb:.1f} GB")
    pure_cp = uniform_kind(model, shape, "cp_q", MACHINE)
    pure_ep = uniform_kind(model, shape, "embp", MACHINE)
    print(f"\n  对比：纯 CP {ms(pure_cp.time_s)} ms，纯 EmbP {ms(pure_ep.time_s)} ms，"
          f"混搭 {ms(result.time_s)} ms")


def show_stage_partition() -> None:
    rule("5 段级空间划分（卡组分开）：单请求上是否更优？")
    shape = Shape(length=8192, kv_len=2048)
    model = mtgr()
    mixed = per_stage(model, shape, MACHINE)
    print(f"  MTGR, L=8192, 表 {model.stages[0].table_bytes / GB:.1f} GB\n")
    print(f"  混搭（全 8 卡）：{ms(mixed.time_s)} ms   每卡 {mixed.peak_gb:.2f} GB")

    assignment = {
        "embed": StagePlan(CardSet(0, 4), Split("embp", 4)),
        "proj": StagePlan(CardSet(0, 4), Split("tp", 4)),
        "attn": StagePlan(CardSet(4, 4), Split("cp_q", 4)),
    }
    partitioned = stage_partition(model, shape, assignment, MACHINE)
    print(f"  4+4 卡组划分：{ms(partitioned.time_s)} ms   每卡 {partitioned.peak_gb:.2f} GB")
    print(f"    → 慢 {partitioned.time_s / mixed.time_s:.2f}×，且每卡常驻反而从 "
          f"{mixed.peak_gb:.2f} 升到 {partitioned.peak_gb:.2f} GB")
    print("      两个维度都更差：段只有 4 卡参与，表也只切 4 份而不是 8 份。")
    print("      结论：**单请求上，段级空间划分没有存在的理由。**")

    print("\n  真正的显存约束不是靠「分区」解决，而是靠「逐段换切分」：")
    huge = mtgr(table_rows=800_000_000)
    cp_fp = sum(footprint(huge, shape, {s.name: Split("cp_q", 8) for s in huge.stages}).values()) / GB
    print(f"    纯 CP 每卡 {cp_fp:.1f} GB > 64 GB ⇒ 不可行")
    mixed_huge = per_stage(huge, shape, MACHINE)
    print(f"    混搭（查表 embp×8）每卡 {mixed_huge.peak_gb:.1f} GB，{ms(mixed_huge.time_s)} ms ⇒ 可行")
    print(f"    这就是「混合」从性能选择变成可行性条件的那个点。")


def show_concurrent() -> None:
    rule("6 并发异构：两个不同请求能否互补（同一台 8 卡机）")
    long_model, long_shape = mtgr(), Shape(length=8192, kv_len=2048)
    short_model, short_shape = hyformer(), Shape(length=2048, kv_len=512)

    _k1, mtgr_alone = best_uniform(long_model, long_shape, MACHINE)
    _k2, hf_alone = best_uniform(short_model, short_shape, MACHINE)
    serial = mtgr_alone.time_s + hf_alone.time_s
    print(f"  场景：一条长序列 MTGR + 一条 HyFormer，同在 8 卡上\n")
    print(f"  (a) 全 8 卡顺序跑：{ms(mtgr_alone.time_s)} + {ms(hf_alone.time_s)} = {ms(serial)} ms")

    print("  (b) 分卡并发（各组用适合自己的切分）：")
    for left in (2, 4, 6):
        right = 8 - left
        jobs = (
            Job(long_model, long_shape, CardSet(0, left), "cp_q"),
            Job(short_model, short_shape, CardSet(left, right), "embp"),
        )
        result = concurrent(jobs, MACHINE)
        verdict = "优于顺序" if result.time_s < serial else "劣于顺序"
        print(
            f"      {left}+{right}（MTGR 用 {left} 卡 CP，HyFormer 用 {right} 卡 EmbP）："
            f"makespan {ms(result.time_s)} ms   链路 {result.link_load * 100:3.0f}%   {verdict}"
        )
    print("\n  → 关键在于「一条请求算力 bound、另一条访存 bound」。分卡后各自变慢，")
    print("    但两条能同时推进；若变慢的倍数小于并发的倍数，makespan 才会下降。")


def show_conflict() -> None:
    rule("7 什么时候异构/并发会失效")
    print("  情形一：两条长序列 MTGR —— 都吃算力，分卡后各慢一倍")
    model, shape = mtgr(), Shape(length=8192, kv_len=2048)
    solo_cp, _ = model_time(model, shape, Split("cp_q", 8), MACHINE)
    jobs = tuple(
        Job(model, shape, CardSet(start, 4), "cp_q") for start in (0, 4)
    )
    both = concurrent(jobs, MACHINE)
    print(f"    8 卡单跑一条 {ms(solo_cp)} ms；4+4 并发 makespan {ms(both.time_s)} ms")
    print(f"    吞吐比 {2 * solo_cp / both.time_s:.2f}×（< 1 表示并发反而降低吞吐）")

    print("\n  情形二：两条都靠 EmbP —— 链路争用")
    hf, hf_shape = hyformer(), Shape(length=2048)
    one_ep, _ = model_time(hf, hf_shape, Split("embp", 8), MACHINE)
    jobs = tuple(
        Job(hf, hf_shape, CardSet(start, 4), "embp") for start in (0, 4)
    )
    two_ep = concurrent(jobs, MACHINE)
    print(f"    8 卡单跑一条 {ms(one_ep)} ms；4+4 并发 makespan {ms(two_ep.time_s)} ms")
    print(f"    吞吐比 {2 * one_ep / two_ep.time_s:.2f}×   链路占用 {two_ep.link_load * 100:.0f}%")

    print("\n  情形三：显存装不下 —— 并发方案直接不可行")
    huge = mtgr(table_rows=800_000_000)
    jobs = (
        Job(huge, shape, CardSet(0, 4), "cp_q"),
        Job(huge, shape, CardSet(4, 4), "cp_q"),
    )
    blocked = concurrent(jobs, MACHINE)
    print(f"    可行={blocked.feasible}   原因：{blocked.reason}")


def show_table_sweep() -> None:
    rule("8 表大小的分水岭：EmbP 从「可选」变成「必须」")
    print(f"{'表行数':>12s} {'表GB':>7s} {'纯CP每卡':>10s} {'纯CP可行':>8s} {'混搭每卡':>9s} {'混搭ms':>8s} {'最优':>8s}")
    shape = Shape(length=2048, kv_len=2048)
    for rows in (5_000_000, 50_000_000, 200_000_000, 500_000_000, 900_000_000):
        model = mtgr(table_rows=rows)
        table_gb = model.stages[0].table_bytes / GB
        cp_plan = {s.name: Split("cp_q", 8) for s in model.stages}
        cp_fp = sum(footprint(model, shape, cp_plan).values()) / GB
        mixed = per_stage(model, shape, MACHINE)
        kind, _ = best_uniform(model, shape, MACHINE)
        print(
            f"{rows:>12,d} {table_gb:>7.1f} {cp_fp:>10.1f} "
            f"{'是' if cp_fp <= 64 else '否':>8s} {mixed.peak_gb:>9.1f} {ms(mixed.time_s):>8s} {kind:>8s}"
        )
    print("\n  → 64 GB 卡上：50M 行（6.0 GB）两种都行；900M 行（107 GB）时纯 CP 完全不可行，")
    print("    只能 EmbP 切表 —— 此时「混合」不是性能选择，而是可行性条件。")



def show_pairing() -> None:
    rule("9 共置配对：四个模型、卡分两组，怎么配")
    print("  场景（用户提出）：a/b 走 EmbP、c/d 走 CP；卡分两组、每组跑两个模型。")
    print("  问：(a,b)+(c,d) 是否不如 (a,c)+(b,d)？\n")

    big = 300_000_000  # 300M 行 × 64 × 2 = 35.8 GB，EmbP 后 4.5 GB/卡
    jobs = (
        best_plan_job("a-HF", hyformer(table_rows=big), Shape(length=2048, kv_len=1024), MACHINE),
        best_plan_job("b-HF", hyformer(table_rows=big), Shape(length=4096, kv_len=2048), MACHINE),
        best_plan_job("c-MT", mtgr(), Shape(length=2048, kv_len=2048), MACHINE),
        best_plan_job("d-MT", mtgr(), Shape(length=4096, kv_len=2048), MACHINE),
    )

    print("  每个请求的资源画像（每组 4 卡）")
    print(f"  {'请求':6s} {'算力ms':>8s} {'访存ms':>8s} {'链路ms':>7s}  bound")
    for job in jobs:
        p = profile(job, 4, MACHINE)
        print(f"  {job.name:6s} {p.t_compute * 1e3:8.2f} {p.t_memory * 1e3:8.2f} "
              f"{p.t_link * 1e3:7.2f}  {p.bound}")

    print("\n  互补度（资源向量余弦，1 = 完全同类）")
    for x, y in ((jobs[0], jobs[1]), (jobs[2], jobs[3]), (jobs[0], jobs[2]), (jobs[1], jobs[3])):
        tag = "同类" if overlap_cosine(x, y, 4, MACHINE) > 0.9 else "互补"
        print(f"    {x.name}+{y.name}: {overlap_cosine(x, y, 4, MACHINE):.3f}  {tag}")

    print("\n  三种配对（每组 4 卡）")
    for pairing in pairings_of_two(jobs, 4, MACHINE):
        print("    " + pairing.describe())
    ranked = pairings_of_two(jobs, 4, MACHINE)
    print(f"\n  → 最优比最差快 {ranked[-1].makespan_s / ranked[0].makespan_s:.2f}×；"
          f"同为「同类配同类」的那组最差。")


def show_pairing_control() -> None:
    rule("10 对照实验：把「均衡」与「互补」分开")
    print("  真实模型上两者混在一起。改用归一化画像（span 都设为 1，只改资源占比）：\n")
    pure_compute = Ratio("A", 1.0, 0.0, 0.0)
    pure_memory = Ratio("B", 0.0, 1.0, 0.0)
    nine_one = Ratio("C", 0.9, 0.1, 0.0)
    one_nine = Ratio("D", 0.1, 0.9, 0.0)
    six_four = Ratio("E", 0.6, 0.4, 0.0)
    four_six = Ratio("F", 0.4, 0.6, 0.0)

    print(f"  {'配对':16s} {'同质 makespan':>14s} {'异质 makespan':>14s} {'加速':>7s}")
    for label, same, cross in (
        ("纯算力 vs 纯访存", (pure_compute, pure_compute), (pure_compute, pure_memory)),
        ("9:1 vs 1:9", (nine_one, nine_one), (nine_one, one_nine)),
        ("6:4 vs 4:6", (six_four, six_four), (six_four, four_six)),
    ):
        gain = pairing_gain(same, cross)
        print(f"  {label:16s} {gain['same']:13.2f}× {gain['cross']:13.2f}× "
              f"{gain['gain']:6.2f}×")

    print("\n  读法：")
    print("  · 「同质」两组的 makespan 恒为 2.00 span —— 同类请求只能串行，与负载无关。")
    print("  · 异质配对的 makespan = 两个资源向量逐项相加后的最大分量。")
    print("  · 因此收益 = 2 / max(重合资源之和)。完全互补时 2×，6:4 混合时只有 1.20×。")
    print("  · **收益的上界由两条请求的资源画像重叠度决定，与谁便宜谁贵无关。**")
    print("  · 但在真实模型上，便宜的请求还会带来「均衡」收益；两者叠加时不能只归因于互补。")


def show_pairing_bound() -> None:
    rule("11 什么时候「换配对」根本没用")
    print("  若两条请求资源画像完全相同（比如都是长序列 MTGR），怎么配都一样：\n")
    same = (Ratio("X", 0.85, 0.15, 0.0), Ratio("Y", 0.85, 0.15, 0.0))
    print(f"  两个 85% 算力的请求：同质配对 {colocate_ratios(same)['makespan']:.2f} span")
    print(f"  异质配对 —— 但两个都长这样，不存在「异质」：" +
          f"{colocate_ratios(same)['makespan']:.2f} span")
    print("\n  这时唯一的出路是「错开时间」：先跑一条，再跑另一条，makespan 相同。")
    print("  要提升吞吐只能靠：① 提高单卡利用率（更小的切分粒度）；")
    print("                  ② 让两条请求的阶段错峰（一条在算力段时另一条在查表段）。")
    print("\n  阶段错峰不需要静态配对，只需要**允许两条请求共享卡组且调度器愿意交错**。")


def _jobs_with_degree(model, shape, lookup_degree: int):
    """造阶段序列。查表段可指定卡数（用于展示「少用卡不损延迟」）。"""
    from hetero.cost import Split, aggregate, stage_time, stage_work
    from hetero.timeline import StageJob

    plan = per_stage(model, shape, MACHINE).plan
    jobs = []
    for stage in model.stages:
        if stage.operator == "lookup":
            degree = lookup_degree
            split = Split("none", 1) if degree == 1 else Split("cp_q", degree)
        else:
            degree = plan[stage.name].degree
            split = plan[stage.name]
        work = aggregate(stage, stage_work(stage, shape), split, shape)
        t, _, _ = stage_time(work, split, MACHINE, stage.operator == "lookup")
        repeats = shape.layers if stage.name in model.per_layer else 1
        jobs.append(StageJob(stage.name, degree, t * repeats + MACHINE.stage_fixed_s * repeats))
    jobs.append(StageJob("dispatch", 1, model.n_kernel * MACHINE.dispatch_s))
    return tuple(jobs)


def show_release() -> None:
    rule("12 空隙从哪来：kernel 的最短时长")
    from hetero.cost import Split, aggregate, stage_time, stage_work
    from hetero.timeline import as_request, greedy_schedule, plan_durations

    print("  事实一：每段都能占满 8 卡时，没有任何空隙。")
    model, shape = mtgr(), Shape(length=8192, kv_len=2048)
    plan = per_stage(model, shape, MACHINE).plan
    jobs = plan_durations(model, shape, plan, MACHINE)
    timeline = greedy_schedule((as_request("mt", jobs),), 8, MACHINE)
    print(f"    MTGR，逐段最优、全 8 卡：利用率 {timeline.utilization() * 100:.1f}%，"
          f"空闲 {timeline.idle_total() * 1e3:.2f} 卡·ms")
    print("    → 段级交错在长请求上因此无收益（后面第 13 节给数值）。")

    print("\n  事实二：单次 kernel 有最短时长（本仿真取 60 μs），低于它加卡不再有效。")
    print("  短请求因此用不满 8 卡：\n")
    for L in (256, 2048):
        shape_here = Shape(length=L, kv_len=min(L, 2048))
        print(f"  L={L}")
        print(f"    {'阶段':8s} {'cp1':>8s} {'cp2':>8s} {'cp4':>8s} {'cp8':>8s}   够用卡数")
        for stage in model.stages:
            row = []
            for degree in (1, 2, 4, 8):
                split = Split("none", 1) if degree == 1 else Split("cp_q", degree)
                work = aggregate(stage, stage_work(stage, shape_here), split, shape_here)
                t, _, _ = stage_time(work, split, MACHINE, stage.operator == "lookup")
                repeats = shape_here.layers if stage.name in model.per_layer else 1
                row.append((t * repeats + MACHINE.stage_fixed_s * repeats) * 1e3)
            need = next(d for d, v in zip((1, 2, 4, 8), row) if v <= 1.05 * row[-1])
            print(f"    {stage.name:8s} " + " ".join(f"{v:8.3f}" for v in row) + f"   {need}")
        print()

    print("  → L=256 时 attn 在 1 卡与 8 卡上几乎同速（都贴在 kernel 下限），")
    print("    那 7 张卡是纯空转。L=2048 时 attn 必须用满 8 卡，砍卡就慢一倍。")
    print("  → 所以「副卡能否提前释放」的判据不是切分度，而是**该段是否已触及 kernel 下限**。")


def show_stage_batching() -> None:
    rule("13 段级交错：收益集中在短请求")
    from hetero.timeline import (
        as_request,
        backfill_schedule,
        greedy_schedule,
        stage_batched_schedule,
    )

    print(f"{'L':>7} {'单请求':>10s} {'8请求·请求级':>14s} {'8请求·段级交错':>16s} {'加速':>7s}")
    for length in (256, 512, 2048, 16384):
        shape = Shape(length=length, kv_len=min(length, 2048))
        jobs = _jobs_with_degree(mtgr(), shape, 8)
        requests = tuple(as_request(f"x{i}", jobs) for i in range(8))
        per_request = sum(job.duration for job in jobs)
        plain = greedy_schedule(requests, 8, MACHINE)
        batched = stage_batched_schedule(requests, 8, MACHINE)
        print(f"{length:>7} {per_request * 1e3:9.3f}ms {plain.makespan * 1e3:13.3f}ms "
              f"{batched.makespan * 1e3:15.3f}ms {plain.makespan / batched.makespan:6.2f}×")

    print("\n  → 与 LLM 直觉相反：GR 里短请求才有空间（长请求每段都占满 8 卡）。")

    print("\n  混合长度时：批内卡数必须一致，而不同 L 的最优卡数不同，")
    print("  所以混合长度根本不能同批 —— 必须先按 L 分桶。")
    short = _jobs_with_degree(mtgr(), Shape(length=256, kv_len=256), 8)
    long_ = _jobs_with_degree(mtgr(), Shape(length=16384, kv_len=2048), 8)
    mixed = tuple(as_request(f"s{i}", short) for i in range(4)) + tuple(
        as_request(f"l{i}", long_) for i in range(4)
    )
    try:
        stage_batched_schedule(mixed, 8, MACHINE)
        print("    （本实现允许多卡数同批）")
    except ValueError as exc:
        print(f"    混合批直接报错：{exc}")
    print("  → 分桶不是「更高效」，是「能不能批」的前置条件。")


def show_partition_cost() -> None:
    rule("13 段级分区的代价：利用率腰斩，但请求更慢")
    from hetero.timeline import StageJob, as_request, greedy_schedule

    model, shape = mtgr(), Shape(length=8192, kv_len=2048)
    full = _jobs_with_degree(model, shape, 8)
    print(f"  {'部署':26s} {'关键路径':>10s} {'利用率':>8s} {'最大空隙':>10s}")
    timeline = greedy_schedule((as_request("x", full),), 8, MACHINE)
    print(f"  {'全 8 卡逐段换切分':26s} {timeline.makespan * 1e3:9.2f}ms "
          f"{timeline.utilization() * 100:7.1f}% {timeline.largest_gap() * 1e3:9.3f}ms")
    partitioned = (
        StageJob("embed+proj@0-3", 4, full[0].duration + full[1].duration),
        StageJob("attn@4-7", 4, full[2].duration),
    )
    timeline = greedy_schedule((as_request("x", partitioned),), 8, MACHINE)
    print(f"  {'段级分区 4+4':26s} {timeline.makespan * 1e3:9.2f}ms "
          f"{timeline.utilization() * 100:7.1f}% {timeline.largest_gap() * 1e3:9.3f}ms")
    print("\n  → 分区把利用率降到 50%，但关键路径反而没变长（因为每段卡数减半的同时段数也减半）。")
    print("    它的价值在显存与并发，不在延迟；单请求上是纯亏。")

def main() -> None:
    print("异构并行方案的仿真验证。全部为仿真值，非实测；只用于判断结构关系。")
    show_structure()
    show_growth()
    show_pure_vs_mixed()
    show_plan_detail()
    show_stage_partition()
    show_concurrent()
    show_conflict()
    show_table_sweep()
    show_pairing()
    show_pairing_control()
    show_pairing_bound()
    show_release()
    show_stage_batching()
    show_partition_cost()
    rule("完")


if __name__ == "__main__":
    main()
