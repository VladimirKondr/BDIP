"""Модуль обучения и оценки классификаторов."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.classifiers import BaseClassifierWrapper
from core.data_manager import DataManager


@dataclass(frozen=True)
class TrainResult:
    """Результат обучения одного классификатора на одном паттерне.

    Attributes:
        pattern_name: название паттерна.
        classifier_name: название классификатора.
        accuracy: доля верных ответов на обучающей выборке.
        Z: сглаженная карта предсказаний на сетке (для визуализации).
        xx: сетка X (meshgrid).
        yy: сетка Y (meshgrid).
    """

    pattern_name: str
    classifier_name: str
    accuracy: float
    Z: NDArray[np.int64]
    xx: NDArray[np.float64]
    yy: NDArray[np.float64]


def train_and_evaluate(
    clf: BaseClassifierWrapper,
    data: DataManager,
    pattern_name: str,
    mesh_step: float = 0.5,
) -> TrainResult:
    """Обучает классификатор и строит карту решений.

    Args:
        clf: обёртка классификатора.
        data: менеджер данных с обучающей выборкой.
        pattern_name: название паттерна (для записи результата).
        mesh_step: шаг сетки для визуализации.

    Returns:
        Заполненный TrainResult.
    """
    X, y = data.X, data.y
    clf.fit(X, y)
    accuracy = clf.score(X, y)

    xx, yy, grid = data.make_mesh(step=mesh_step, margin=5.0)
    Z = clf.predict(grid).reshape(xx.shape)

    return TrainResult(
        pattern_name=pattern_name,
        classifier_name=clf.name,
        accuracy=accuracy,
        Z=Z,
        xx=xx,
        yy=yy,
    )
