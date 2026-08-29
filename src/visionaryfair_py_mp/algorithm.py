import numpy as np
from numpy.typing import NDArray

from src.visionaryfair_py_mp import structs


def compute(
    pool: list[structs.ItemMeta],
    histories: list[structs.DrawHistory],
    settings: structs.WeightSettings,
    batch_size: int = 1,
) -> structs.WeightResult:
    if not pool:
        raise ValueError(f'候选池为空 ({pool=}); 周期已抽干, 应由调用方重置计数。')
    if batch_size > len(pool):
        raise ValueError(f'批量 {batch_size} 超过池内人数 {len(pool)}; 上限可能已把池子抽小')

    pool_size = len(pool)
    draw_count_by_id = {r.id: r.count for r in histories}
    draw_count = np.zeros(pool_size, np.double)
    multiplier_sum = 0.0

    for i in range(pool_size):
        item = pool[i]

        if item.multiplier <= 0:
            raise ValueError(f'项目 {item.id} 的倍率 {item.multiplier} 必须为正的有限值')

        count = draw_count_by_id.get(item.id, 0)
        if count <= 0:
            raise ValueError(f'项目 {item.id} 的次数为负')

        draw_count[i] = count
        multiplier_sum += item.multiplier

    total_draws = draw_count.sum()
    share = np.array([s.multiplier for s in pool], np.double) / multiplier_sum

    personal_horizon = settings.personal_horizon * pool_size
    personal_debt = np.zeros(pool_size, np.double)
    weight = np.zeros(pool_size, np.double)
    for i in range(pool_size):
        personal_debt[i] = max(
            share[i] * (total_draws + personal_horizon) - draw_count[i], 0
        )
        weight[i] = personal_debt[i]

    dimension_count = len(settings.dimensions)
    dimension_debt = np.zeros((pool_size, dimension_count), np.double)
    for slot in range(dimension_count):
        dimension = settings.dimensions[slot]
        if dimension.dimension_id < 0:
            raise ValueError(
                f'维度 `{settings.dimensions[slot].dimension_id=} ({slot=})` 下标不能为负'
            )

        label_count = 0
        for i in range(pool_size):
            labels = pool[i].labels
            if dimension.dimension_id >= len(labels):
                raise ValueError(
                    f'项目 {pool[i].id} 缺少维度 {dimension.dimension_id} 的标签'
                )

            label = labels[dimension.dimension_id]
            if label < 0:
                raise ValueError(f'项目 {pool[i].id} 的标签为负')

            label_count = max(label_count, label + 1)

        label_share = np.zeros(label_count)
        label_drawn = np.zeros(label_count)
        for i in range(pool_size):
            label = pool[i].labels[dimension.dimension_id]
            label_share[label] += share[i]
            label_drawn[label] += draw_count[i]

        horizon = dimension.horizon_per_pick * batch_size
        debt = np.amax(
            [
                label_share * (total_draws + horizon) - label_drawn,
                np.zeros(label_count),
            ],
            axis=0,
        )
        for i in range(pool_size):
            val = debt[pool[i].labels[dimension.dimension_id]]
            dimension_debt[i][slot] = val
            weight[i] *= val

    weight_sum = weight.sum()
    degraded = weight_sum <= 0
    if degraded:
        weight = np.ones(pool_size, np.double)
        weight_sum = pool_size
    candidates = [
        structs.CandidateWeight(
            id=pool[i].id,
            weight=weight[i],
            debt=personal_debt[i],
            dimension_debts=dimension_debt[i],
        )
        for i in range(pool_size)
    ]

    return structs.WeightResult(
        candidates, weight_sum, bool(degraded), pool_size == 1
    )


def to_probabilities(
    result: structs.WeightResult, settings: structs.WeightSettings
) -> NDArray[np.double]:
    floor = settings.lowest_rnd
    if floor < 0 or floor > 1:
        raise ValueError(f'{floor=} 必须在 [0, 1] 内')

    pool_size = len(result.candidates)
    floor_share = floor / pool_size
    probabilities = np.array(pool_size)

    for i in range(pool_size):
        probabilities[i] = (1 - floor) * result.candidates[
            i
        ].weight / result.weight_sum + floor_share

    return probabilities
