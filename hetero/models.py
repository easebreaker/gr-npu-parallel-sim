"""四个推荐模型的阶段图，与源码结构对齐。

仿真不追求绝对时间，只要求**结构关系正确**：哪一段随什么增长、
主导资源是谁、能不能 EmbP、切分能削多少。这些都由源码定，不由参数定。

对齐来源：
- `执行时间预测/timepred/recipes.py` 的四套配方
- `执行时间预测/文档/算子复杂度-按算子手册.md` 各算子章
- `_vendor/hstu_probe.py` 对官方 HSTU 单层的形状核对

一条重要的结构差异（决定了「谁适合 CP」）：

| 模型 | 长度依赖 | 由什么决定 |
|---|---|---|
| MTGR / HSTU | **L²** | 整段自注意力，分数矩阵 B·h·L² |
| LONGER | **L·T** | 组内注意力 R=⌊L/T⌋，每组长 T —— 对 L 是线性 |
| HyFormer | **几乎无** | decode 是 short×short；seq_cross 的 KV 是 kv_len；boost 固定 |
"""

from __future__ import annotations

from dataclasses import dataclass, field

FULL = 1.0


@dataclass(frozen=True)
class Stage:
    """一个阶段的结构量。全部由源码定。

    - ``tokens``：linear 类算子的 token 轴
    - ``seq``：attention 类算子的 Q/KV 长度来源
    - ``out_width``：linear 的输出宽度（uvqk 合计宽度，不是隐藏维）
    """

    name: str
    operator: str
    growth: str  # "lookup" / "L2" / "L" / "LxT" / "const" / "l_kv"
    tokens: str = "real"  # real / group / short / boost
    seq: str = "length"  # length / group_tokens / short / kv
    out_width: float = 0.0
    table_bytes: float = 0.0
    weight_bytes: float = 0.0
    comm: str = "none"  # none / alltoall / allgather
    note: str = ""


@dataclass(frozen=True)
class Shape:
    batch: int = 1
    length: int = 2048
    short_len: int = 64
    kv_len: int = 512
    boost: int = 16
    group_tokens: int = 16
    d: int = 256
    hidden: int = 1024
    heads: int = 8
    d_head: int = 32
    layers: int = 4
    elem: int = 2

    @property
    def n_tok(self) -> int:
        return self.batch * self.length

    @property
    def groups(self) -> int:
        return max(1, self.length // self.group_tokens)

    def tokens_of(self, axis: str) -> float:
        if axis == "real":
            return float(self.n_tok)
        if axis == "group":
            return float(self.batch * self.groups * self.group_tokens)
        if axis == "short":
            return float(self.batch * self.short_len)
        if axis == "boost":
            return float(self.batch * self.boost)
        raise KeyError(axis)

    def seq_of(self, axis: str) -> float:
        if axis == "length":
            return float(self.length)
        if axis == "group_tokens":
            return float(self.group_tokens)
        if axis == "short":
            return float(self.short_len)
        if axis == "kv":
            return float(self.kv_len)
        raise KeyError(axis)


@dataclass(frozen=True)
class Model:
    name: str
    stages: tuple[Stage, ...]
    n_kernel: int = 3
    per_layer: tuple[str, ...] = field(default_factory=tuple)
    note: str = ""

    def stage(self, name: str) -> Stage:
        for item in self.stages:
            if item.name == name:
                return item
        raise KeyError(name)


def _uvqk_width(hidden: int, heads: int, d_head: int) -> float:
    """官方 HSTU 把 u、v、q、k 合成一个权重：2·hidden·h + 2·d_head·h。"""
    return 2.0 * hidden * heads + 2.0 * d_head * heads


def mtgr(table_rows: int = 50_000_000) -> Model:
    """MTGR / HSTU：查表 → uvqk 投影 → 整段自注意力（掩码）。"""
    return Model(
        name="MTGR/HSTU",
        stages=(
            Stage(
                "embed",
                "lookup",
                growth="lookup",
                table_bytes=table_rows * 64 * 2,
                comm="alltoall",
                note="表巨大，是唯一需要 EmbP 的理由",
            ),
            Stage(
                "proj",
                "linear",
                growth="L",
                tokens="real",
                out_width=_uvqk_width(1024, 8, 32),
                comm="allgather",
                note="uvqk 四投影合一；另有一次 o 投影，这里并入 out_width",
            ),
            Stage(
                "attn",
                "self_attn",
                growth="L2",
                seq="length",
                comm="alltoall",
                note="掩码让候选互不可见，但稠密实现仍算整段 L²",
            ),
        ),
        n_kernel=3,
        per_layer=("proj", "attn"),
        note="四模型里唯一有 L² 的，CP 收益最大",
    )


def longer(table_rows: int = 50_000_000) -> Model:
    """LONGER：查表 → 组内注意力 → 组内 FFN → 混合交叉 → 短序列注意力。"""
    return Model(
        name="LONGER",
        stages=(
            Stage("embed", "lookup", growth="lookup", table_bytes=table_rows * 64 * 2, comm="alltoall"),
            Stage(
                "inner_attn",
                "group_attn",
                growth="LxT",
                seq="group_tokens",
                comm="none",
                note="R=⌊L/T⌋，每组长 T ⇒ 总工作量 ∝ L·T，对 L 是线性",
            ),
            Stage("inner_ffn", "linear", growth="L", tokens="group", out_width=4096, comm="none"),
            Stage(
                "hybrid_cross",
                "cross_attn",
                growth="l_kv",
                seq="kv",
                comm="alltoall",
                note="N_q 是全局 token + 少量近期行为（很小），KV 是整段历史 ∝ L",
            ),
            Stage(
                "short_attn",
                "self_attn",
                growth="const",
                seq="short",
                comm="none",
                note="只读 short_len，与用户序列无关",
            ),
        ),
        n_kernel=5,
        per_layer=("inner_attn", "inner_ffn", "hybrid_cross", "short_attn"),
        note="没有 L²，CP 只能拿到 L 的线性收益",
    )


def hyformer(table_rows: int = 50_000_000) -> Model:
    """HyFormer（longer 编码器）：查表 → 序列侧交叉 → decode → boost。"""
    return Model(
        name="HyFormer",
        stages=(
            Stage("embed", "lookup", growth="lookup", table_bytes=table_rows * 64 * 2, comm="alltoall"),
            Stage(
                "seq_cross",
                "cross_attn",
                growth="l_kv",
                seq="kv",
                comm="alltoall",
                note="Q=short_len，KV=kv_len；kv_len 通常被截断到固定上限",
            ),
            Stage(
                "seq_ffn",
                "linear",
                growth="const",
                tokens="short",
                out_width=1024,
                comm="none",
                note="按 short_len 的 token 数",
            ),
            Stage(
                "decode",
                "cross_attn",
                growth="const",
                seq="short",
                comm="alltoall",
                note="Q 与 KV 都是 short_len ⇒ 与用户序列长度无关",
            ),
            Stage("boost_mix", "permute", growth="const", tokens="boost", comm="none", note="boost 固定的小常数"),
            Stage("boost_ffn", "linear", growth="const", tokens="boost", out_width=1024, comm="none"),
        ),
        n_kernel=6,
        per_layer=("seq_cross", "seq_ffn", "decode", "boost_mix", "boost_ffn"),
        note="整网对用户序列长度几乎不敏感，CP 几乎无用",
    )


ALL_MODELS = {"mtgr": mtgr, "longer": longer, "hyformer": hyformer}
