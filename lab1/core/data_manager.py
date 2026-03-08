"""Менеджер данных: хранение точек обучающей выборки."""

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray


@dataclass
class DataManager:
    """Хранит точки двух (или более) классов в 2D пространстве.

    Attributes:
        points: массив координат точек (N, 2).
        labels: массив меток классов (N,), от 0 до num_classes-1.
    """

    points: list[list[float]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)

    def add_point(self, x: float, y: float, class_id: int) -> None:
        """Добавляет точку с меткой класса.

        Args:
            x: координата X.
            y: координата Y.
            class_id: метка класса (0, 1, ...).
        """
        self.points.append([x, y])
        self.labels.append(class_id)

    def clear(self) -> None:
        """Очищает все накопленные точки."""
        self.points.clear()
        self.labels.clear()

    def load_arrays(
        self, points: NDArray[np.float64], labels: NDArray[np.int64]
    ) -> None:
        """Загружает данные из numpy-массивов.

        Args:
            points: массив формы (N, 2).
            labels: массив меток формы (N,).
        """
        self.clear()
        self.points = points.tolist()
        self.labels = labels.tolist()

    @property
    def X(self) -> NDArray[np.float64]:
        """Возвращает матрицу признаков (N, 2)."""
        return np.array(self.points, dtype=np.float64)

    @property
    def y(self) -> NDArray[np.int64]:
        """Возвращает вектор меток (N,)."""
        return np.array(self.labels, dtype=np.int64)

    @property
    def num_classes(self) -> int:
        """Количество уникальных классов в текущих данных."""
        return len(set(self.labels)) if self.labels else 0

    @property
    def is_ready(self) -> bool:
        """True, если данных достаточно для обучения (>= 2 класса, >= 4 точки)."""
        return self.num_classes >= 2 and len(self.points) >= 4

    def make_mesh(
        self, step: float = 0.5, margin: float = 5.0
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Строит равномерную сетку по диапазону данных для визуализации.

        Args:
            step: шаг сетки.
            margin: отступ за пределы данных.

        Returns:
            Тройка (xx, yy, Z_flat), где xx, yy — координатные сетки,
            Z_flat — вытянутый массив точек для predict.
        """
        X = self.X
        x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
        y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, step),
            np.arange(y_min, y_max, step),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        return xx, yy, grid
