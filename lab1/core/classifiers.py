"""Реестр классификаторов scikit-learn с единым интерфейсом."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class BaseClassifierWrapper(ABC):
    """Абстрактный обёртка над sklearn-классификатором.

    Attributes:
        name: человекочитаемое название классификатора.
        model: экземпляр sklearn-классификатора.
    """

    name: str

    def __init__(self) -> None:
        self.model: ClassifierMixin = self._build_model()

    @abstractmethod
    def _build_model(self) -> ClassifierMixin:
        """Создаёт и возвращает sklearn-модель."""

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> None:
        """Обучает модель.

        Args:
            X: матрица признаков (N, 2).
            y: вектор меток (N,).
        """
        self.model.fit(X, y)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Предсказывает метки классов.

        Args:
            X: матрица признаков (M, 2).

        Returns:
            Массив предсказанных меток (M,).
        """
        return self.model.predict(X)  # type: ignore[return-value]

    def score(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
    ) -> float:
        """Возвращает долю верных предсказаний (accuracy).

        Args:
            X: матрица признаков.
            y: истинные метки.

        Returns:
            Значение accuracy в диапазоне [0, 1].
        """
        return float(self.model.score(X, y))


# ---------------------------------------------------------------------------
# Конкретные обёртки
# ---------------------------------------------------------------------------


class NBCWrapper(BaseClassifierWrapper):
    name = "NBC (Наивный Байес)"

    def _build_model(self) -> ClassifierMixin:
        return GaussianNB()


class SVMWrapper(BaseClassifierWrapper):
    name = "SVM (RBF-ядро)"

    def _build_model(self) -> ClassifierMixin:
        return SVC(kernel="rbf", C=1.0, gamma="scale", probability=False)


class SVMLinearWrapper(BaseClassifierWrapper):
    name = "SVM (линейное)"

    def _build_model(self) -> ClassifierMixin:
        return SVC(kernel="linear", C=1.0, probability=False)


class KNNWrapper(BaseClassifierWrapper):
    name = "kNN (k=5)"

    def _build_model(self) -> ClassifierMixin:
        return KNeighborsClassifier(n_neighbors=5)


class DTWrapper(BaseClassifierWrapper):
    name = "Дерево решений"

    def _build_model(self) -> ClassifierMixin:
        return DecisionTreeClassifier(max_depth=5, random_state=42)


class AdaBoostWrapper(BaseClassifierWrapper):
    name = "AdaBoost"

    def _build_model(self) -> ClassifierMixin:
        return AdaBoostClassifier(n_estimators=100, random_state=42)


class GBTWrapper(BaseClassifierWrapper):
    name = "GBT (Gradient Boosting)"

    def _build_model(self) -> ClassifierMixin:
        return GradientBoostingClassifier(n_estimators=100, random_state=42)


class RFWrapper(BaseClassifierWrapper):
    name = "Random Forest"

    def _build_model(self) -> ClassifierMixin:
        return RandomForestClassifier(n_estimators=100, random_state=42)


class ETWrapper(BaseClassifierWrapper):
    name = "Extremely Random Trees"

    def _build_model(self) -> ClassifierMixin:
        return ExtraTreesClassifier(n_estimators=100, random_state=42)


class ANNWrapper(BaseClassifierWrapper):
    name = "ANN (MLP)"

    def _build_model(self) -> ClassifierMixin:
        return MLPClassifier(
            hidden_layer_sizes=(64, 64),
            max_iter=1000,
            random_state=42,
        )


class LDAWrapper(BaseClassifierWrapper):
    name = "LDA"

    def _build_model(self) -> ClassifierMixin:
        return LinearDiscriminantAnalysis()


def get_all_classifiers() -> list[BaseClassifierWrapper]:
    """Возвращает список экземпляров всех доступных классификаторов.

    Returns:
        Список обёрток над sklearn-классификаторами.
    """
    return [
        NBCWrapper(),
        SVMLinearWrapper(),
        SVMWrapper(),
        KNNWrapper(),
        DTWrapper(),
        AdaBoostWrapper(),
        GBTWrapper(),
        RFWrapper(),
        ETWrapper(),
        ANNWrapper(),
        LDAWrapper(),
    ]
