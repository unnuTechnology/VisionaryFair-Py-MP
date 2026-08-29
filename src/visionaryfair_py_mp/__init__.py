import warnings
from dataclasses import dataclass
from functools import cached_property
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from src.visionaryfair_py_mp import algorithm, structs

T = TypeVar


@dataclass
class Item:
    """
    用于表示一个项目，存储源对象、ID、元信息与历史信息。
    """

    val: object
    id: int
    meta: structs.ItemMeta
    history: structs.DrawHistory

    def __init__(
        self,
        val: T,
        id: int,
        multiplier: float = 1,
        max_pick: int = 0,
        labels: NDArray[np.int32] | None = None,
    ):
        """
        初始化一个项目。

        Args:
            val (T): 源对象。
            id (int): 对象唯一 ID。
            multiplier (float, optional): 爆率倍率。基准为默认值 1.
            max_pick (int, optional): 单轮次中最高抽取次数。为 0 时表示不指定。
            labels (NDArray[np.int32] | None, optional): 需要做均衡的分类维度，用整数。例如：[1, 2, 3] 代表在维度 0 数值为 1，在第 1 维度数值为 2，在第 2 维度数值为 3。为 None 时将初始化一个空数组。
        """
        if labels is None:
            labels = np.array([], np.int32)

        self.val = val
        self.id = id
        self.meta = structs.ItemMeta(id, multiplier, max_pick, labels)
        self.history = structs.DrawHistory(id, 0)


class VisionaryFairModel:
    """使用 VisionaryFairMP (VFMP) 算法的独立随机抽取类"""

    def __init__(
        self,
        items: list[Item] | None = None,
        settings: structs.WeightSettings | None = None,
    ):
        """
        构建一个 VFMP 抽取类。

        Args:
            items (list[Item] | None, optional): 抽取池内的项目。为 None 时将会构建一个空列表。
            settings (structs.WeightSettings | None, optional): 算法相关设置。为 None 时将会使用 structs.WeightSettings 的默认设置。
        """
        if items is None:
            items = []
        if settings is None:
            settings = structs.WeightSettings()

        self.items, self.settings = items, settings

    def pick(self, size: int) -> list[Item]:
        wres = algorithm.compute(
            self.available_item_metas,
            self.available_item_histories,
            self.settings,
            size,
        )
        probs = algorithm.to_probabilities(wres, self.settings)
        res_idxes = np.random.default_rng().choice(
            len(self.available_items), size=size, p=probs
        )
        results = [self.available_items[int(i)] for i in res_idxes]
        for item in results:
            item.history.count += 1
        try:
            del self.available_items
        except AttributeError:
            warnings.warn('Failed to refresh available_items', RuntimeWarning)

        return results

    def reset_history(self) -> None:
        """
        重设所有项目抽取次数记录。
        """
        for item in self.items:
            item.history.count = 0

    @property
    def item_metas(self) -> list[structs.ItemMeta]:
        """
        所有 Item 的 meta 属性组成的列表。
        """
        return [item.meta for item in self.items]

    @property
    def item_histories(self) -> list[structs.DrawHistory]:
        """
        所有 Item 的 history 属性组成的列表。
        """
        return [item.history for item in self.items]

    @cached_property
    def available_items(self) -> list[Item]:
        """
        仍可被抽选的项目列表。
        """
        return [i for i in self.items if self._available_items_filter(i)]

    @property
    def available_item_metas(self) -> list[structs.ItemMeta]:
        """
        所有可用 Item 的 meta 属性组成的列表。
        """
        return [item.meta for item in self.available_items]

    @property
    def available_item_histories(self) -> list[structs.DrawHistory]:
        """
        所有可用 Item 的 history 属性组成的列表。
        """
        return [item.history for item in self.available_items]

    @staticmethod
    def _available_items_filter(item: Item) -> bool:
        """
        用于筛选仍可用的项目。

        Args:
            item (Item): 要筛选的项目。

        Returns:
            bool: 是否可用。
        """
        if item.meta.max_pick == 0:
            return True
        else:
            return item.history.count < item.meta.max_pick


__all__ = ('Item', 'VisionaryFairModel', 'algorithm', 'structs')
