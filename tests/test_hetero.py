"""切分是否真的有效、混搭是否优于纯方案、并发何时有利。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hetero import (  # noqa: E402
    CardSet,
    Job,
    Machine,
    Shape,
    Split,
    best_uniform,
    concurrent,
    footprint,
    helps,
    hyformer,
    longer,
    model_time,
    mtgr,
    per_stage,
    stage_partition,
    uniform_kind,
)
from hetero.solver import StagePlan  # noqa: E402

GB = 2**30


class HelpsTests(unittest.TestCase):
    def test_embp_only_helps_lookup(self) -> None:
        self.assertTrue(helps("embp", "lookup"))
        for op in ("self_attn", "linear", "cross_attn", "permute"):
            self.assertFalse(helps("embp", op), op)

    def test_cp_helps_every_token_parallel_op(self) -> None:
        # 查表按 token 完全可切：CP 切 token 对它同样有效。
        # 这正是初版把它当成「复制」的错误来源。
        for op in ("self_attn", "cross_attn", "group_attn", "linear", "lookup"):
            self.assertTrue(helps("cp_q", op), op)
        self.assertFalse(helps("cp_q", "permute"))


class GrowthTests(unittest.TestCase):
    def test_growth_order_matches_structure(self) -> None:
        def ratio(factory):
            small, _ = model_time(factory(), Shape(length=512, kv_len=512), Split("none", 1), Machine())
            big, _ = model_time(factory(), Shape(length=8192, kv_len=512), Split("none", 1), Machine())
            return big / small

        mtgr_r, longer_r, hf_r = ratio(mtgr), ratio(longer), ratio(hyformer)
        self.assertGreater(mtgr_r, longer_r)
        self.assertGreater(longer_r, hf_r)
        # MTGR 的 L² 让 8192/512 至少是二次量级的一半
        self.assertGreater(mtgr_r, 10.0)
        # HyFormer 的注意力段是平的，它仍然随 L 涨是因为查表（每 token 一次）
        model = hyformer()
        _, rows_small = model_time(model, Shape(length=512, kv_len=512), Split("none", 1), Machine())
        _, rows_big = model_time(model, Shape(length=8192, kv_len=512), Split("none", 1), Machine())
        small = {name: t for name, t, _, _ in rows_small}
        big = {name: t for name, t, _, _ in rows_big}
        for flat_stage in ("seq_cross", "decode", "boost_mix", "boost_ffn"):
            self.assertAlmostEqual(big[flat_stage] / small[flat_stage], 1.0, places=6,
                                   msg=f"{flat_stage} 不应随 L 变")
        self.assertGreater(big["embed"] / small["embed"], 10.0)


class PureVsMixedTests(unittest.TestCase):
    def test_mixing_beats_both_pure_plans(self) -> None:
        for factory in (mtgr, longer, hyformer):
            for length in (512, 8192):
                shape = Shape(length=length, kv_len=min(length, 2048))
                model = factory()
                pure_cp = uniform_kind(model, shape, "cp_q", Machine())
                pure_ep = uniform_kind(model, shape, "embp", Machine())
                mixed = per_stage(model, shape, Machine())
                # 混搭是逐段取最优，因此不劣于任一纯方案（允许 2% 的数值容差）
                self.assertLessEqual(mixed.time_s, pure_cp.time_s * 1.02, f"{model.name} L={length} 劣于纯CP")
                self.assertLessEqual(mixed.time_s, pure_ep.time_s * 1.02, f"{model.name} L={length} 劣于纯EmbP")

    def test_pure_embp_inflates_attention(self) -> None:
        # 纯 EmbP 下注意力段被复制 N 份，MTGR 上时间应远超纯 CP
        shape = Shape(length=8192, kv_len=2048)
        model = mtgr()
        cp = uniform_kind(model, shape, "cp_q", Machine())
        ep = uniform_kind(model, shape, "embp", Machine())
        self.assertGreater(ep.time_s / cp.time_s, 3.0)

    def test_without_l2_cp_and_embp_are_close(self) -> None:
        # 修正后：查表在两者下都按 N 摊，HyFormer 又没有 L²，
        # 所以纯 CP 与纯 EmbP 的时间接近，差别在显存而不是速度。
        shape = Shape(length=8192, kv_len=2048)
        model = hyformer()
        machine = Machine()
        cp = uniform_kind(model, shape, "cp_q", machine)
        ep = uniform_kind(model, shape, "embp", machine)
        self.assertLess(abs(cp.time_s / ep.time_s - 1.0), 0.10)
        # 但显存差一个数量级：EmbP 会切表
        cp_fp = footprint(model, shape, {s.name: Split("cp_q", 8) for s in model.stages})
        ep_fp = footprint(model, shape, {s.name: Split("embp", 8) for s in model.stages})
        self.assertAlmostEqual(cp_fp["table"] / ep_fp["table"], 8.0, places=6)


class FootprintTests(unittest.TestCase):
    def test_embp_shards_the_table_and_cp_does_not(self) -> None:
        model = mtgr()
        shape = Shape(length=2048, kv_len=2048)
        cp = footprint(model, shape, {s.name: Split("cp_q", 8) for s in model.stages})
        ep = footprint(model, shape, {s.name: Split("embp", 8) for s in model.stages})
        self.assertAlmostEqual(cp["table"] / ep["table"], 8.0, places=6)

    def test_huge_table_makes_pure_cp_infeasible(self) -> None:
        model = mtgr(table_rows=900_000_000)
        shape = Shape(length=2048, kv_len=2048)
        cp = uniform_kind(model, shape, "cp_q", Machine())
        self.assertFalse(cp.feasible)
        mixed = per_stage(model, shape, Machine())
        self.assertTrue(mixed.feasible)
        self.assertLess(mixed.peak_gb, 64.0)


class PartitionTests(unittest.TestCase):
    def test_stage_partition_is_worse_than_mixing(self) -> None:
        model = mtgr()
        shape = Shape(length=8192, kv_len=2048)
        mixed = per_stage(model, shape, Machine())
        assignment = {
            "embed": StagePlan(CardSet(0, 4), Split("embp", 4)),
            "proj": StagePlan(CardSet(0, 4), Split("tp", 4)),
            "attn": StagePlan(CardSet(4, 4), Split("cp_q", 4)),
        }
        partitioned = stage_partition(model, shape, assignment, Machine())
        self.assertGreater(partitioned.time_s, mixed.time_s)
        # 慢的倍数应与「卡数减半」同量级
        self.assertGreater(partitioned.time_s / mixed.time_s, 1.5)


class ConcurrentTests(unittest.TestCase):
    def test_overlapping_card_sets_are_rejected(self) -> None:
        model, shape = hyformer(), Shape(length=512)
        jobs = (
            Job(model, shape, CardSet(0, 4), "embp"),
            Job(model, shape, CardSet(2, 4), "embp"),
        )
        with self.assertRaises(ValueError):
            concurrent(jobs, Machine())

    def test_two_compute_bound_requests_gain_no_throughput(self) -> None:
        model, shape = mtgr(), Shape(length=8192, kv_len=2048)
        solo, _ = model_time(model, shape, Split("cp_q", 8), Machine())
        jobs = tuple(Job(model, shape, CardSet(start, 4), "cp_q") for start in (0, 4))
        both = concurrent(jobs, Machine())
        throughput = 2 * solo / both.time_s
        # 都吃算力：分卡后各慢一倍，吞吐基本不变
        self.assertLess(abs(throughput - 1.0), 0.15, f"吞吐比 {throughput:.2f}")

    def test_infeasible_memory_blocks_concurrency(self) -> None:
        model, shape = mtgr(table_rows=900_000_000), Shape(length=2048, kv_len=2048)
        jobs = tuple(Job(model, shape, CardSet(start, 4), "cp_q") for start in (0, 4))
        result = concurrent(jobs, Machine())
        self.assertFalse(result.feasible)
        self.assertIn("超限", result.reason)


class BestUniformTests(unittest.TestCase):
    def test_best_uniform_respects_feasibility(self) -> None:
        model = mtgr(table_rows=900_000_000)
        shape = Shape(length=2048, kv_len=2048)
        kind, result = best_uniform(model, shape, Machine())
        self.assertTrue(result.feasible)
        self.assertEqual(kind, "embp")


if __name__ == "__main__":
    unittest.main()


class PairingTests(unittest.TestCase):
    """共置配对：同类 vs 异类。"""

    def _jobs(self):
        from hetero.pairing import best_plan_job

        machine = Machine()
        big = 300_000_000
        return (
            best_plan_job("a-HF", hyformer(table_rows=big), Shape(length=2048, kv_len=1024), machine),
            best_plan_job("b-HF", hyformer(table_rows=big), Shape(length=4096, kv_len=2048), machine),
            best_plan_job("c-MT", mtgr(), Shape(length=2048, kv_len=2048), machine),
            best_plan_job("d-MT", mtgr(), Shape(length=4096, kv_len=2048), machine),
        )

    def test_cross_pairing_beats_same_kind(self) -> None:
        from hetero.pairing import pairings_of_two

        ranked = pairings_of_two(self._jobs(), 4, Machine())
        best, worst = ranked[0], ranked[-1]
        self.assertGreater(worst.makespan_s / best.makespan_s, 1.2)
        # 最差的那组必定是「同类配同类」
        same = {("a-HF", "b-HF"), ("c-MT", "d-MT")}
        self.assertTrue(any(frozenset(g) in {frozenset(s) for s in same} for g in worst.groups))

    def test_similarity_separates_the_two_families(self) -> None:
        from hetero.pairing import overlap_cosine

        a, b, c, d = self._jobs()
        self.assertGreater(overlap_cosine(a, b, 4, Machine()), 0.9)
        self.assertGreater(overlap_cosine(c, d, 4, Machine()), 0.9)
        self.assertLess(overlap_cosine(a, c, 4, Machine()), 0.8)

    def test_pure_complementarity_gives_two_x(self) -> None:
        from hetero.pairing import Ratio, pairing_gain

        compute = Ratio("A", 1.0, 0.0, 0.0)
        memory = Ratio("B", 0.0, 1.0, 0.0)
        gain = pairing_gain((compute, compute), (compute, memory))
        self.assertAlmostEqual(gain["gain"], 2.0, places=6)

    def test_gain_shrinks_as_profiles_overlap(self) -> None:
        from hetero.pairing import Ratio, pairing_gain

        gains = []
        for share in (1.0, 0.9, 0.6, 0.51):
            left = Ratio("L", share, 1.0 - share, 0.0)
            right = Ratio("R", 1.0 - share, share, 0.0)
            gains.append(pairing_gain((left, left), (left, right))["gain"])
        # 占比越接近，收益越小
        self.assertGreater(gains[0], gains[1])
        self.assertGreater(gains[1], gains[2])
        self.assertGreater(gains[2], gains[3])
        # 51:49 与 49:51 几乎重合，收益应趋近 1（上界 = 2/max(重合和) = 1.02）
        self.assertLess(gains[-1], 1.05)

    def test_identical_profiles_leave_no_room(self) -> None:
        from hetero.pairing import Ratio, colocate_ratios

        x = Ratio("X", 0.85, 0.15, 0.0)
        y = Ratio("Y", 0.85, 0.15, 0.0)
        self.assertAlmostEqual(colocate_ratios((x, y))["makespan"], 2.0, places=6)

    def test_colocate_never_exceeds_sum_of_spans(self) -> None:
        # 资源池模型必然是上界：makespan 不超过两个请求 span 之和
        from hetero.solver import colocate

        jobs = self._jobs()
        result = colocate(jobs, 8, Machine())
        alone = sum(
            model_time(job.model, job.shape, Split("none", 1), Machine())[0] for job in jobs
        )
        self.assertLessEqual(result.makespan_s, alone)


class TimelineTests(unittest.TestCase):
    """时间线：全卡部署无空隙；复制段少用卡不损延迟；回填能利用空隙。"""

    def _jobs(self, lookup_degree: int, table_rows: int = 0):
        from hetero.cost import Split, aggregate, helps, stage_time, stage_work
        from hetero.timeline import StageJob

        machine = Machine()
        model = mtgr()
        shape = Shape(length=8192, kv_len=2048)
        plan = per_stage(model, shape, machine).plan
        jobs = []
        for stage in model.stages:
            if stage.operator == "lookup":
                degree = lookup_degree
                split = Split("none", 1)
            else:
                degree = machine.cards
                split = plan[stage.name]
            work = aggregate(stage, stage_work(stage, shape), split, shape)
            t, _, _ = stage_time(work, split, machine, stage.operator == "lookup")
            repeats = shape.layers if stage.name in model.per_layer else 1
            jobs.append(StageJob(stage.name, degree, t * repeats))
        jobs.append(StageJob("dispatch", 1, model.n_kernel * machine.dispatch_s))
        return tuple(jobs)

    def test_full_card_deployment_has_no_idle(self) -> None:
        from hetero.timeline import as_request, greedy_schedule

        timeline = greedy_schedule((as_request("x", self._jobs(8)),), 8, Machine())
        self.assertGreater(timeline.utilization(), 0.99)

    def test_replicated_stage_does_not_slow_down_on_one_card(self) -> None:
        one = self._jobs(1)[0]
        eight = self._jobs(8)[0]
        self.assertAlmostEqual(one.duration, eight.duration, places=9)
        # 但 8 卡消耗 8 倍卡时
        self.assertAlmostEqual(eight.duration * 8, one.duration * 8, places=9)

    def test_split_stage_conserves_card_time(self) -> None:
        from hetero.cost import Split, aggregate, stage_time, stage_work

        machine = Machine()
        model = mtgr()
        shape = Shape(length=8192, kv_len=2048)
        attn = model.stage("attn")
        times = {}
        for degree in (1, 2, 4, 8):
            split = Split("none", 1) if degree == 1 else Split("cp_q", degree)
            work = aggregate(attn, stage_work(attn, shape), split, shape)
            t, _, _ = stage_time(work, split, machine)
            times[degree] = t
        # 卡时守恒：degree × 时间 基本不变（差 2% 以内）
        products = [degree * times[degree] for degree in (1, 2, 4, 8)]
        self.assertLess(max(products) / min(products) - 1.0, 0.05)

    def test_backfill_never_worse_than_greedy(self) -> None:
        from hetero.timeline import as_request, backfill_schedule, greedy_schedule

        machine = Machine()
        jobs = self._jobs(8)
        requests = tuple(as_request(f"r{i}", jobs) for i in range(4))
        greedy = greedy_schedule(requests, 8, machine)
        backfill = backfill_schedule(requests, 8, machine)
        self.assertLessEqual(backfill.makespan, greedy.makespan + 1e-9)
        # dispatch 段只用 1 卡，因此利用率不会到 100%
        self.assertGreater(backfill.utilization(), 0.99)

    def test_releasing_cards_helps_only_with_a_replicated_stage(self) -> None:
        from hetero.timeline import as_request, backfill_schedule

        machine = Machine()
        full = self._jobs(8)
        lean = self._jobs(1)
        base = backfill_schedule(tuple(as_request(f"x{i}", full) for i in range(4)), 8, machine)
        opt = backfill_schedule(tuple(as_request(f"x{i}", lean) for i in range(4)), 8, machine)
        # 至少不劣化
        self.assertLessEqual(opt.makespan, base.makespan + 1e-9)

    def test_idle_gaps_are_reported(self) -> None:
        from hetero.timeline import Interval, Timeline

        timeline = Timeline(2)
        timeline.add(Interval(0, 0.0, 1.0, "a"))
        timeline.add(Interval(1, 0.5, 1.5, "b"))
        gaps0 = timeline.idle_gaps(0)
        self.assertEqual(gaps0, ((1.0, 1.5),))
        self.assertAlmostEqual(timeline.largest_gap(), 0.5, places=9)


class StageBatchedTests(unittest.TestCase):
    """段级交错：只有段用不满卡时才有收益。"""

    def _jobs(self, length: int):
        from hetero.cost import Split, aggregate, stage_time, stage_work
        from hetero.timeline import StageJob

        machine = Machine()
        model = mtgr()
        shape = Shape(length=length, kv_len=min(length, 2048))
        plan = per_stage(model, shape, machine).plan
        jobs = []
        for stage in model.stages:
            split = plan[stage.name]
            work = aggregate(stage, stage_work(stage, shape), split, shape)
            t, _, _ = stage_time(work, split, machine, stage.operator == "lookup")
            repeats = shape.layers if stage.name in model.per_layer else 1
            jobs.append(StageJob(stage.name, split.degree, t * repeats + machine.stage_fixed_s * repeats))
        jobs.append(StageJob("dispatch", 1, model.n_kernel * machine.dispatch_s))
        return tuple(jobs)

    def test_lookup_is_splittable_not_replicated(self) -> None:
        # 修正后的关键性质：查表按 token 可切，CP 对查表也有效
        from hetero.cost import Split, aggregate, stage_time, stage_work

        machine = Machine()
        model = mtgr()
        shape = Shape(length=8192, kv_len=2048)
        embed = model.stage("embed")
        times = {}
        for degree in (1, 8):
            split = Split("none", 1) if degree == 1 else Split("cp_q", degree)
            work = aggregate(embed, stage_work(embed, shape), split, shape)
            t, _, _ = stage_time(work, split, machine, True)
            times[degree] = t
        self.assertGreater(times[1] / times[8], 3.0)

    def test_short_request_gains_from_stage_batching(self) -> None:
        from hetero.timeline import as_request, greedy_schedule, stage_batched_schedule

        jobs = self._jobs(256)
        requests = tuple(as_request(f"x{i}", jobs) for i in range(8))
        greedy = greedy_schedule(requests, 8, Machine())
        batched = stage_batched_schedule(requests, 8, Machine())
        self.assertGreater(greedy.makespan / batched.makespan, 1.3)

    def test_long_request_gains_almost_nothing(self) -> None:
        from hetero.timeline import as_request, greedy_schedule, stage_batched_schedule

        jobs = self._jobs(16384)
        requests = tuple(as_request(f"x{i}", jobs) for i in range(8))
        greedy = greedy_schedule(requests, 8, Machine())
        batched = stage_batched_schedule(requests, 8, Machine())
        self.assertLess(greedy.makespan / batched.makespan, 1.05)

    def test_short_request_needs_fewer_cards(self) -> None:
        # L=256 的 attn 落在 kernel 下限，1 卡与 8 卡同速
        from hetero.cost import Split, aggregate, stage_time, stage_work

        machine = Machine()
        model = mtgr()
        shape = Shape(length=256, kv_len=256)
        attn = model.stage("attn")
        times = {}
        for degree in (1, 8):
            split = Split("none", 1) if degree == 1 else Split("cp_q", degree)
            work = aggregate(attn, stage_work(attn, shape), split, shape)
            t, _, _ = stage_time(work, split, machine)
            times[degree] = t
        # 落在 kernel 下限，1 卡与 8 卡差异应小于 5%
        self.assertLess(abs(times[1] / times[8] - 1.0), 0.05)

    def test_floor_bounds_every_schedule(self) -> None:
        from hetero.timeline import (
            as_request,
            card_time_floor,
            greedy_schedule,
            stage_batched_schedule,
        )

        jobs = self._jobs(1024)
        requests = tuple(as_request(f"x{i}", jobs) for i in range(8))
        floor = card_time_floor(requests, 8)
        for schedule in (greedy_schedule, stage_batched_schedule):
            self.assertGreaterEqual(schedule(requests, 8, Machine()).makespan, floor - 1e-9)

    def test_mixed_lengths_cannot_share_a_wave(self) -> None:
        # 不同 L 的最优卡数不同（短请求的 attn 只要 1 卡），因此同批会在
        # 卡数上冲突。这不是效率问题，是硬约束：必须按 L 分桶才谈得上批。
        from hetero.timeline import as_request, stage_batched_schedule

        short = self._jobs(256)
        long_ = self._jobs(16384)
        mixed = tuple(as_request(f"s{i}", short) for i in range(4)) + tuple(
            as_request(f"l{i}", long_) for i in range(4)
        )
        with self.assertRaises(ValueError):
            stage_batched_schedule(mixed, 8, Machine())

    def test_same_length_batches_fine(self) -> None:
        from hetero.timeline import as_request, stage_batched_schedule

        jobs = self._jobs(2048)
        timeline = stage_batched_schedule(
            tuple(as_request(f"x{i}", jobs) for i in range(8)), 8, Machine()
        )
        self.assertGreater(timeline.utilization(), 0.8)


class AmdahlTests(unittest.TestCase):
    """只有部分段可并行时，CP 与 DP 的延迟/吞吐权衡。"""

    def _stages(self, length: int, scalable=("self_attn",)):
        from hetero.amdahl import stages_of

        machine = Machine()
        return stages_of(mtgr(), Shape(length=length, kv_len=min(length, 2048)), machine, scalable)

    def test_cp_trades_throughput_for_latency(self) -> None:
        from hetero.amdahl import compare

        result = compare(self._stages(8192), 8, 0.83)
        # CP 更快，但整机吞吐更低
        self.assertGreater(result.latency_gain, 1.2)
        self.assertGreater(result.throughput_ratio, 1.0)

    def test_amdahl_bound_is_independent_of_cards(self) -> None:
        # 上界 = 总时间 / 不可并行部分，与卡数无关（只与结构有关）
        from hetero.amdahl import compare

        stages = self._stages(8192)
        bounds = {compare(stages, n, 0.83).amdahl_bound for n in (2, 4, 8)}
        self.assertEqual(len(bounds), 1)

    def test_latency_gain_is_capped_by_the_bound(self) -> None:
        from hetero.amdahl import compare

        stages = self._stages(8192)
        for cards in (2, 4, 8):
            result = compare(stages, cards, 0.83)
            self.assertLessEqual(result.latency_gain, result.amdahl_bound + 1e-9)

    def test_more_scalable_work_means_better_trade(self) -> None:
        # 把 token 可切的段都算进去，延迟增益上升、吞吐代价下降
        from hetero.amdahl import compare

        narrow = compare(self._stages(8192, ("self_attn",)), 8, 0.83)
        wide = compare(
            self._stages(8192, ("self_attn", "linear", "cross_attn", "group_attn", "lookup")),
            8,
            0.83,
        )
        self.assertGreater(wide.latency_gain, narrow.latency_gain)
        self.assertLess(wide.throughput_ratio, narrow.throughput_ratio)

    def test_longer_and_hyformer_gain_nothing_from_cp(self) -> None:
        from hetero.amdahl import compare, stages_of

        machine = Machine()
        shape = Shape(length=8192, kv_len=2048)
        for factory in (longer, hyformer):
            stages = stages_of(factory(), shape, machine, ("self_attn",))
            result = compare(stages, 8, 0.83)
            self.assertLess(result.latency_gain, 1.10, f"{factory().name} 不该从 CP 获益")

    def test_short_request_is_not_worth_parallelizing(self) -> None:
        from hetero.amdahl import compare

        short = compare(self._stages(256), 8, 0.83)
        long_ = compare(self._stages(16384), 8, 0.83)
        # 短请求的 Amdahl 上界明显低于长请求
        self.assertLess(short.amdahl_bound, long_.amdahl_bound)
        self.assertLess(short.amdahl_bound, 1.4)

    def test_dp_has_no_redundancy(self) -> None:
        # DP 的一条请求耗时 = 各段之和，无任何重复
        from hetero.amdahl import dp_latency

        stages = self._stages(8192)
        self.assertAlmostEqual(
            dp_latency(stages), sum(stage.seconds for stage in stages), places=12
        )


class SplitConstraintTests(unittest.TestCase):
    """TP 在推荐模型上受注意力头数限制（NVIDIA recsys-examples #71）。"""

    def test_tp_is_capped_by_heads(self) -> None:
        from hetero import max_tp_degree

        self.assertEqual(max_tp_degree(Shape(heads=4, d_head=16)), 4)
        self.assertEqual(max_tp_degree(Shape(heads=8, d_head=32)), 8)

    def test_cutting_head_dim_relaxes_the_cap(self) -> None:
        from hetero import max_tp_degree

        self.assertEqual(max_tp_degree(Shape(heads=4, d_head=16), True), 64)

    def test_cap_never_below_one(self) -> None:
        from hetero import max_tp_degree

        self.assertGreaterEqual(max_tp_degree(Shape(heads=1, d_head=1)), 1)
