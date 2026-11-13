"""切分的边际成本：b(N) 的形状识别与最优并行度的数值求解。

对应文档 `并行度内点-方案.md`。核心是把「并行度越大越快」这条**假设**
换成一条**可测量的曲线**，并承认它的形状未知。

三段判断，顺序不能换：

1. **形状**（`fit_shape`）：b(N) 随 N 怎么长。三种候选（线性 / 次线性 / 饱和）
   给出完全不同的结论，必须先定它。
2. **内点**（`optimal_degree`）：在给定形状下，净收益的极值点。
   不假设线性——直接数值求。
3. **饱和区**（`is_saturated`）：该段工作量是否已低于 kernel 下限。
   落在这里时，任何切分都只是增加 b。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DEGREES = (1, 2, 4, 6, 8)


@dataclass(frozen=True)
class ShapeFit:
    """b(N) 的拟合结果。`kind` 为 linear / sqrt / log / saturated。"""

    kind: str
    b0: float
    b1: float
    rms: float
    residual_sign_changes: int
    n_points: int

    @property
    def monotone(self) -> bool:
        return self.b1 > 0

    def at(self, n: float) -> float:
        if self.kind == "linear":
            return self.b0 + self.b1 * n
        if self.kind == "sqrt":
            return self.b0 + self.b1 * math.sqrt(n)
        if self.kind == "log":
            return self.b0 + self.b1 * math.log(1.0 + n)
        if self.kind == "saturated":
            return self.b0 + self.b1 * (1.0 - math.exp(-n / 4.0))
        raise KeyError(self.kind)

    def summary(self) -> str:
        return (
            f"{self.kind:9s} b0={self.b0 * 1e3:7.2f} ms  b1={self.b1 * 1e3:6.3f}  "
            f"rms={self.rms * 1e3:5.3f} ms  残差变号 {self.residual_sign_changes} 次"
        )


def _fit_linear(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0.0:
        return my, 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - slope * mx, slope


def _basis(kind: str, n: float) -> float:
    if kind == "linear":
        return n
    if kind == "sqrt":
        return math.sqrt(n)
    if kind == "log":
        return math.log(1.0 + n)
    if kind == "saturated":
        return 1.0 - math.exp(-n / 4.0)
    raise KeyError(kind)


_SHAPES = ("linear", "sqrt", "log", "saturated")


def fit_shape(points: list[tuple[float, float]]) -> tuple[ShapeFit, ...]:
    """对每个候选形状做最小二乘，按 rms 升序返回。

    `points` 为 [(N, b)]，b 单位秒。少于 4 点时形状不可辨识——
    两点只能定一条直线，会给出虚假的「线性」。
    """
    if len(points) < 4:
        raise ValueError("形状识别至少需要 4 个不同并行度（推荐 5 个：1,2,4,6,8）")
    xs = [float(n) for n, _ in points]
    ys = [float(b) for _, b in points]
    if len(set(xs)) != len(xs):
        raise ValueError("并行度不能重复")
    out: list[ShapeFit] = []
    for kind in _SHAPES:
        basis = [_basis(kind, x) for x in xs]
        b0, b1 = _fit_linear(basis, ys)
        residuals = [y - (b0 + b1 * b) for y, b in zip(ys, basis)]
        rms = math.sqrt(sum(r * r for r in residuals) / len(residuals))
        signs = 0
        for prev, cur in zip(residuals, residuals[1:]):
            if prev * cur < 0:
                signs += 1
        out.append(ShapeFit(kind, b0, b1, rms, signs, len(points)))
    out.sort(key=lambda item: item.rms)
    return tuple(out)


def identify_shape(points: list[tuple[float, float]], tol: float = 1.0) -> ShapeFit:
    """取 rms 最小的形状；若最优与前一名差距不到 `tol` 倍，判为不可辨识。

    直接在返回值上体现：`kind` 会是 "ambiguous"，此时不应继续求内点。
    """
    fits = fit_shape(points)
    best, runner_up = fits[0], fits[1]
    if runner_up.rms <= best.rms * tol:
        return ShapeFit(
            "ambiguous",
            best.b0,
            best.b1,
            best.rms,
            best.residual_sign_changes,
            best.n_points,
        )
    return best


def grow_rate(fit: ShapeFit, n: float) -> float:
    """在 N 处的边际增长率 db/dN（秒/卡）。

    这是内点判据里与 `ρ·C·N^(-ρ-1)` 相比较的量。
    """
    eps = 1e-4
    return (fit.at(n + eps) - fit.at(n - eps)) / (2 * eps)


def optimal_degree(
    fit: ShapeFit,
    compute_latency: float,
    rho: float = 0.83,
    upper: float = 64.0,
    steps: int = 6400,
) -> tuple[float, float]:
    """净收益极值点。返回 (N*, 该点总时间)。

    目标：`T(N) = b(N) + C·N^(-rho)`。
    不假设 b 的形状——用 `fit.at` 数值求。
    """
    if compute_latency <= 0:
        raise ValueError("计算项要为正")
    best_n, best_t = 1.0, fit.at(1.0) + compute_latency
    for index in range(1, steps + 1):
        n = 1.0 + (upper - 1.0) * index / steps
        t = fit.at(n) + compute_latency * n ** (-rho)
        if t < best_t:
            best_n, best_t = n, t
    return best_n, best_t


def penalty_of(fit: ShapeFit, at_n: int, compute_latency: float, rho: float = 0.83) -> float:
    """在 `at_n` 张卡上运行，比最优并行度贵多少（相对值）。"""
    _, best = optimal_degree(fit, compute_latency, rho)
    here = fit.at(float(at_n)) + compute_latency * float(at_n) ** (-rho)
    return here / best - 1.0


# ---------------------------------------------------------------- 饱和区


@dataclass(frozen=True)
class Saturation:
    """某一段在不同并行度下的耗时，以及是否已触及 kernel 下限。"""

    name: str
    times: dict[int, float]
    floor: float

    def ratio(self) -> float:
        """8 卡耗时 / 1 卡耗时。≈1 表示切分无效。"""
        one = self.times.get(1)
        eight = self.times.get(8)
        if not one or not eight:
            raise ValueError("需要 N=1 与 N=8 两个点")
        return eight / one

    @property
    def saturated(self) -> bool:
        """8 卡不比 1 卡快 10% 以上，即判饱和。"""
        return self.ratio() > 0.90

    def needed_degree(self) -> int:
        """达到 8 卡方案 1.05 倍以内的最小卡数。"""
        best = self.times.get(8)
        if best is None:
            raise ValueError("需要 N=8")
        for degree in sorted(self.times):
            if self.times[degree] <= best * 1.05:
                return degree
        return max(self.times)


def is_saturated(times: dict[int, float], floor: float | None = None) -> bool:
    """便捷判据：8 卡相对 1 卡的加速不到 1.11 倍，即落进饱和区。"""
    one = times.get(1)
    eight = times.get(8)
    if not one or not eight:
        raise ValueError("需要 N=1 与 N=8")
    return one / eight < 1.11

# ---------------------------------------------------------------- 边际斜率（推荐判据）


def slope_at(
    fit: ShapeFit,
    n: float = 8.0,
    delta: float = 0.02,
) -> float:
    """拟合曲线在 N 处的斜率（秒/卡）。

    这是内点判据真正需要的量：把 `b'(N)` 与 `rho·C·N^(-rho-1)` 相比。
    比「认出形状的标签」稳健得多——形状标签在噪声下常判错，
    但斜率在同样噪声下的偏差小于 5%（见 `tests/test_marginal.py`）。
    """
    return (fit.at(n + delta) - fit.at(n - delta)) / (2.0 * delta)


def inner_point_exists(
    fit: ShapeFit,
    compute_latency: float,
    rho: float = 0.83,
    upper: float = 64.0,
) -> bool:
    """内点是否落在 `[1, upper]` 内。

    必须**扫全区间**，不能在某个固定 N 处比较 `b'(N)` 与边际节省。
    反例：b 线性、C 较小时，N=8 处 `b'` 远大于边际节省（看起来「已过内点」），
    但真正的最优其实在 N≈2 —— 因为在 N=1 附近边际节省远大于 `b'`。
    只在一个点上比较会同时误判两个方向。
    """
    best_n, _ = optimal_degree(fit, compute_latency, rho, upper=upper)
    if best_n <= 1.0 + 1e-9:
        return False
    # 极值在边界（upper）上不算内点：会继续增大
    return best_n < upper - 1e-9


def optimal_degree_within(
    fit: ShapeFit,
    compute_latency: float,
    cards: int,
    rho: float = 0.83,
) -> tuple[int, float, float]:
    """在「本机最多 `cards` 张卡」的约束下取最优档位。

    返回 (最优档位, 该档总时间, 用满 `cards` 相对最优的损失)。
    档位只在 1/2/4/.../cards 里取——真实部署不会用奇数卡。
    """
    if cards < 1:
        raise ValueError("卡数至少 1")
    degrees = [1]
    d = 1
    while d * 2 <= cards:
        d *= 2
        degrees.append(d)
    scored = [(n, fit.at(n) + compute_latency * n ** (-rho)) for n in degrees]
    best_n, best_t = min(scored, key=lambda item: item[1])
    full_t = fit.at(float(cards)) + compute_latency * float(cards) ** (-rho)
    return best_n, best_t, full_t / best_t - 1.0


def saturation_penalty(
    times: dict[int, float],
    best_degree: int | None = None,
) -> float:
    """用 8 卡而非最优卡数时，相对损失是多少。

    若 `times` 里 8 卡已经是最快的，返回 0。
    """
    if 8 not in times:
        raise ValueError("需要 N=8 的耗时")
    fastest = min(times.values())
    return times[8] / fastest - 1.0
