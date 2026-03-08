"""Генераторы трёх типов пространственных паттернов точек."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.data_manager import DataManager

RNG = np.random.default_rng(seed=42)


@dataclass(frozen=True)
class Pattern:
    """Описание одного паттерна.

    Attributes:
        name: краткое название.
        description: словесное описание типа разделимости.
        data: менеджер данных с точками.
    """

    name: str
    description: str
    data: DataManager


# ---------------------------------------------------------------------------
# Паттерн 1: линейно разделимый
# ---------------------------------------------------------------------------


def make_linear_pattern(n_per_class: int = 150) -> Pattern:
    """Два облака точек, разделённых гиперплоскостью.

    Args:
        n_per_class: число точек в каждом классе.

    Returns:
        Pattern с данными.
    """
    c0 = RNG.multivariate_normal(mean=[-15.0, 0.0], cov=[[25, 4], [4, 25]], size=n_per_class)
    c1 = RNG.multivariate_normal(mean=[15.0, 0.0], cov=[[25, 4], [4, 25]], size=n_per_class)

    points = np.vstack([c0, c1])
    labels = np.array([0] * n_per_class + [1] * n_per_class, dtype=np.int64)

    dm = DataManager()
    dm.load_arrays(points, labels)
    return Pattern(
        name="Линейно разделимый",
        description="Два компактных облака, разделённых прямой.",
        data=dm,
    )


# ---------------------------------------------------------------------------
# Паттерн 2: линейно НЕ разделимый — «извилистая» граница
# ---------------------------------------------------------------------------


def make_wavy_pattern(n_per_class: int = 200) -> Pattern:
    """Два класса, разделённых синусоидальной границей.

    Класс 0 — точки ниже y = 12*sin(x/10), класс 1 — выше.

    Args:
        n_per_class: приблизительное число точек в каждом классе.

    Returns:
        Pattern с данными.
    """
    pts: list[list[float]] = []
    labs: list[int] = []

    collected = {0: 0, 1: 0}
    while min(collected.values()) < n_per_class:
        x = RNG.uniform(-50.0, 50.0)
        y = RNG.uniform(-40.0, 40.0)
        boundary = 12.0 * np.sin(x / 10.0)
        cls = 0 if y < boundary else 1
        if collected[cls] < n_per_class:
            pts.append([x, y])
            labs.append(cls)
            collected[cls] += 1

    dm = DataManager()
    dm.load_arrays(np.array(pts, dtype=np.float64), np.array(labs, dtype=np.int64))
    return Pattern(
        name="Линейно неразделимый (sin-граница)",
        description="Граница классов — синусоида. Линейный классификатор справится плохо.",
        data=dm,
    )


# ---------------------------------------------------------------------------
# Паттерн 3: сильно перекрывающиеся классы
# ---------------------------------------------------------------------------


def make_overlapping_pattern(n_per_class: int = 200) -> Pattern:
    """Два класса с существенным перекрытием — нет чёткой границы.

    Args:
        n_per_class: число точек в каждом классе.

    Returns:
        Pattern с данными.
    """
    cov = [[200, 60], [60, 200]]
    c0 = RNG.multivariate_normal(mean=[-5.0, -5.0], cov=cov, size=n_per_class)
    c1 = RNG.multivariate_normal(mean=[5.0, 5.0], cov=cov, size=n_per_class)

    points = np.vstack([c0, c1])
    labels = np.array([0] * n_per_class + [1] * n_per_class, dtype=np.int64)

    dm = DataManager()
    dm.load_arrays(points, labels)
    return Pattern(
        name="Пересекающиеся классы",
        description="Широкие облака с большим перекрытием. Высокая ошибка Байеса.",
        data=dm,
    )


def get_all_patterns() -> list[Pattern]:
    """Возвращает список всех трёх паттернов.

    Returns:
        Паттерны: линейный, синусоидальный, перекрывающийся.
    """
    return [
        make_linear_pattern(),
        make_wavy_pattern(),
        make_overlapping_pattern(),
    ]
