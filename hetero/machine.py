"""机器与链路模型。全部数值是仿真假设，不是实测。

取 910B3 量级：8 卡单机，全互联 HCCS，单卡 64 GB。
有效效率刻意压低（cube 25%、HBM 35%），因为现有实测反解出的
``η`` 比标称低一到两个数量级；这里取偏保守值，让结论不依赖乐观假设。
"""

from __future__ import annotations

from dataclasses import dataclass

GB = 2**30


@dataclass(frozen=True)
class Machine:
    cards: int = 8
    # 单卡能力（标称）
    cube_flops: float = 245e12 / 8
    hbm_bw: float = 1.6e12 / 8
    hbm_cap: float = 64 * GB
    # 卡间链路：HCCS 全互联，这里的数字是「单卡对其他卡的总带宽」
    hcss_bw: float = 56e9
    # 有效效率
    cube_eff: float = 0.25
    hbm_eff: float = 0.35
    hcss_eff: float = 0.60
    # 随机访存的效率惩罚：查表是随机 gather，同一 Warp 内相邻线程访问不同行，
    # 访存无法合并。官方口径下合并前后的差距可达数倍，这里取 0.15（约 6.7 倍惩罚）。
    gather_eff: float = 0.15
    # 下发开销：每 kernel 的固定成本
    dispatch_s: float = 8e-6
    # 每段每层的固定开销（Tiling + Launch + 同步），**与卡数和 L 都无关**。
    # 实测里小段有 90% 的时间属于这一类（讲稿 S18：L=256 时 b 占 90%）。
    # 取 30 μs 是为了让 L=256 的单层 attn 量级与实测一致（约 0.1–0.2 ms/段）。
    stage_fixed_s: float = 30e-6
    # 单次 kernel 的**最短时长**：工作量再小也要这么久（下发 + 同步 + 尾块）。
    # 这条是「小请求填不满机器」的物理根源——低于它之后，加卡不会更快，
    # 正确做法是少用卡、同时跑更多请求。
    # 实测旁证：13 点里 L=256 的 cp2 为 9.75 ms，而计算项只有 0.02 ms 量级。
    min_stage_s: float = 60e-6

    def p_eff(self) -> float:
        return self.cube_flops * self.cube_eff

    def bw_eff(self) -> float:
        return self.hbm_bw * self.hbm_eff

    def link_eff(self) -> float:
        return self.hcss_bw * self.hcss_eff

    def gather_bw(self) -> float:
        """随机 gather 的等效带宽。"""
        return self.bw_eff() * self.gather_eff


@dataclass(frozen=True)
class CardSet:
    """一组卡。异构方案里每一组是一段连续编号，互不重叠。"""

    start: int
    count: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.count < 1:
            raise ValueError("卡组编号与卡数都要为正")
        if self.start + self.count > 8:
            raise ValueError(f"卡组越界：{self.start}+{self.count} > 8")

    def __str__(self) -> str:
        return f"[{self.start}..{self.start + self.count - 1}]"
