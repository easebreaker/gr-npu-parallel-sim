"""并行度内点的判定逻辑：形状识别的陷阱、斜率的稳健性、扫描的必要性。"""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hetero.marginal import (  # noqa: E402
    fit_shape,
    identify_shape,
    inner_point_exists,
    is_saturated,
    optimal_degree,
    optimal_degree_within,
    slope_at,
)

B0, B1 = 7.63, 0.586

SHAPES = {
    "linear": lambda n: B0 + B1 * n,
    "sqrt": lambda n: B0 + B1 * math.sqrt(n),
    "saturated": lambda n: B0 + B1 * (1 - math.exp(-n / 4)),
}

POINTS = (1, 2, 3, 4, 6, 8, 12, 16, 24)


def points_of(name: str, sigma: float = 0.0, seed: int = 0) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    f = SHAPES[name]
    return [(float(n), (f(n) + rng.gauss(0, sigma)) * 1e-3) for n in POINTS]


class ShapeTests(unittest.TestCase):
    def test_three_points_cannot_identify_shape(self) -> None:
        # 三点拟合两参数，0 自由度；本实现直接拒绝
        with self.assertRaises(ValueError):
            fit_shape([(2.0, 8.8e-3), (4.0, 10.7e-3), (8.0, 12.3e-3)])

    def test_noiseless_shapes_are_recovered(self) -> None:
        for name in ("linear", "sqrt"):
            fit = identify_shape(points_of(name))
            self.assertEqual(fit.kind, name, name)
            self.assertLess(fit.rms, 1e-9)

    def test_noiseless_saturated_is_recovered(self) -> None:
        # 无噪声时能认出
        self.assertEqual(identify_shape(points_of("saturated")).kind, "saturated")

    def test_shape_label_fails_under_noise_but_slope_survives(self) -> None:
        # 这是本方案把判据从「形状标签」改成「斜率」的核心理由：
        # σ=0.3 ms 时形状标签会判成 linear（真值是 saturated），完全判错；
        # 但同一批数据的 b'(8) 仍然准确。
        pts = points_of("saturated", sigma=0.3, seed=5)
        fit = fit_shape(pts)[0]
        self.assertNotEqual(fit.kind, "saturated")          # 标签判错
        expected = 0.586 * 0.03
        self.assertLess(abs(slope_at(fit, 8.0) * 1e3 - expected), 0.10)  # 斜率仍准


class SlopeTests(unittest.TestCase):
    def test_slope_separates_shapes_under_heavy_noise(self) -> None:
        # σ=0.5 ms、单次测量，仍能把 0.59 / 0.10 / 0.02 分开
        for name, expected in (("linear", B1), ("sqrt", B1 * 0.17), ("saturated", B1 * 0.03)):
            pts = points_of(name, sigma=0.5, seed=11)
            fit = fit_shape(pts)[0]
            got = slope_at(fit, 8.0) * 1e3
            self.assertLess(abs(got - expected), 0.08, f"{name}: {got:.3f} vs {expected:.3f}")

    def test_linear_slope_is_flat(self) -> None:
        fit = identify_shape(points_of("linear"))
        values = [slope_at(fit, n) for n in (2.0, 4.0, 8.0)]
        self.assertLess(max(values) - min(values), 1e-6)

    def test_sqrt_slope_decays(self) -> None:
        fit = identify_shape(points_of("sqrt"))
        self.assertGreater(slope_at(fit, 2.0), slope_at(fit, 8.0) * 1.5)


class InnerPointTests(unittest.TestCase):
    def test_single_point_check_would_mislead(self) -> None:
        # N=8 处 b' > 边际节省，但最优点在 N≈2.4
        fit = identify_shape(points_of("linear"))
        c = 3.40e-3
        marginal_saving = 0.83 * c * 8.0 ** (-1.83)
        self.assertGreater(slope_at(fit, 8.0), marginal_saving)
        n_star, _ = optimal_degree(fit, c, 0.83, upper=8.0)
        self.assertLess(n_star, 3.0)

    def test_saturation_has_no_inner_point(self) -> None:
        fit = identify_shape(points_of("saturated"))
        self.assertFalse(inner_point_exists(fit, 72.93e-3, upper=64.0))

    def test_long_request_wants_all_cards(self) -> None:
        fit = identify_shape(points_of("linear"))
        for c in (72.93e-3, 143.73e-3):
            _, _, penalty = optimal_degree_within(fit, c, cards=8)
            self.assertLess(penalty, 0.01)

    def test_short_request_pays_for_all_cards(self) -> None:
        fit = identify_shape(points_of("linear"))
        best, _, penalty = optimal_degree_within(fit, 1.40e-3, cards=8)
        self.assertEqual(best, 2)
        self.assertGreater(penalty, 0.25)

    def test_mid_request_is_between(self) -> None:
        fit = identify_shape(points_of("linear"))
        _, _, penalty = optimal_degree_within(fit, 3.40e-3, cards=8)
        self.assertGreater(penalty, 0.15)
        self.assertLess(penalty, 0.30)

    def test_within_prefers_powers_of_two(self) -> None:
        fit = identify_shape(points_of("linear"))
        for c in (1.40e-3, 3.40e-3, 72.93e-3):
            best, _, _ = optimal_degree_within(fit, c, cards=8)
            self.assertIn(best, (1, 2, 4, 8))


class SaturationTests(unittest.TestCase):
    def test_flat_scaling_is_saturated(self) -> None:
        self.assertTrue(is_saturated({1: 0.36e-3, 8: 0.36e-3}))

    def test_linear_scaling_is_not(self) -> None:
        self.assertFalse(is_saturated({1: 2.36e-3, 8: 0.43e-3}))

    def test_needs_both_endpoints(self) -> None:
        with self.assertRaises(ValueError):
            is_saturated({8: 1e-3})


if __name__ == "__main__":
    unittest.main()
