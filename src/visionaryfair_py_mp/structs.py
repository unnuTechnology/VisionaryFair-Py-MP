from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class StudentMeta:
    """
    学生的静态属性。

    Arguments:
        id: 学生唯一标识。
        multiplier: 爆率倍率：1.0 = 基准，2.0 = 长期被点频率翻倍。通过 ``share = Multiplier ÷ ΣMultiplier`` 进入欠账模型，连续可调、长期精确生效。
        max_pick: 本周期学生最多被抽到的次数，作为安全阀。为 ``0`` 代表不限次数。
        labels: 需要做均衡的分类维度，用整数。例如：[1, 2, 3] 代表在维度 0 数值为 1，在第 1 维度数值为 2，在第 2 维度数值为 3。
    """

    id: int
    multiplier: float
    max_pick: int = 0
    labels: NDArray[np.int32] = field(default=np.array([], dtype=np.int32))


class DimensionBalance(NamedTuple):
    """
    维度均衡值配置。

    Arguments:
        dimension_id: 维度编号，对应 ``StudentMeta.labels`` 的下标。
        horizon_per_pick: 视野系数：乘以批量后得到该层的宽容度。
    """

    dimension_id: int
    horizon_per_pick: float = 0.8


@dataclass
class DrawHistory:
    """
    某个学生在本周期内的历史统计。

    Arguments:
        id: 学生唯一标识。
        count: 已被抽中次数。
    """

    id: int
    count: int = 0


@dataclass(frozen=True)
class WeightSettings:
    """
    权重配置。

    Arguments:
        personal_horizon: 个人视野 = 该值 * 池内人数。系统假装未来还要再抽这么多次来算每人应得份额。小 → 谁超支就压很久, 接近点名册; 大 → 允许运气波动, 接近纯随机。个人偏离的稳态方差约等于 该值 / (2*(1 - lowest_rnd)), 硬上界为该值。
        lowest_rnd: 保底份额。任何池内学生的最低被抽概率 = 该值 / 池内人数。它也是交替率的精确调节器: P(A) = 1/2 + (1-e)² / (2(2-e)), 误差 ±0.002。注意池很小时保底会变大 (剩 10 人时每人 1%), 此时权重的影响被稀释。
        dimensions: 需要做均衡的维度。空数组 = 只做个人层均衡。
    """

    personal_horizon: float = 2
    lowest_rnd: float = 0.1
    dimensions: list[DrawHistory] = field(default_factory=list)


@dataclass
class CandidateWeight:
    """
    单个学生的权重附加中间量。

    Arguments:
        id: 学生唯一标识。
        weight: 未归一化的权重。
        debt: 个人欠账: 到未来 H 次为止应得的份额减去已拿到的, 负数归零。
        dimension_debts: 各维度欠账, 与 ``WeightSettings.dimensions`` 同序。
    """

    id: int
    weight: float
    debt: float
    dimension_debts: NDArray[np.double] = field(
        default=np.array([], dtype=np.double)
    )


@dataclass
class WeightResult:
    """
    最终权重配置。

    Arguments:
        candidates: 各学生的最终权重与中间量。
        weight_sum: 权重合计；退化时等于池内人数。
        is_uniformed: 表示全池欠账是否已经同时清零, 并退化为等权。
        is_determined: 表示池内是否只剩一人, 抽取结果确定。
    """

    candidates: list[CandidateWeight]
    weight_sum: float
    is_uniformed: bool
    is_determined: bool
