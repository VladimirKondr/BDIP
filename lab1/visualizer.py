"""Визуализация: сохранение карт решений и сводных таблиц."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from core.trainer import TrainResult
from patterns import Pattern

matplotlib.use("Agg")  # без GUI

# Цвета фона и точек для 2 классов
_BG_COLORS = ["#FFDEDE", "#DEDEFF"]
_PT_COLORS = ["#CC0000", "#0000CC"]


def plot_decision_boundary(
    result: TrainResult,
    pattern: Pattern,
    out_path: Path,
) -> None:
    """Сохраняет график разделяющей поверхности + точки обучающей выборки.

    Args:
        result: результат обучения (содержит сетку предсказаний).
        pattern: паттерн с исходными точками.
        out_path: путь для сохранения .png.
    """
    cmap_bg = ListedColormap(_BG_COLORS[: pattern.data.num_classes])
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.contourf(result.xx, result.yy, result.Z, alpha=0.6, cmap=cmap_bg)

    X = pattern.data.X
    y = pattern.data.y
    for cls_id, color in enumerate(_PT_COLORS[: pattern.data.num_classes]):
        mask = y == cls_id
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            c=color,
            s=12,
            edgecolors="k",
            linewidths=0.3,
            label=f"Класс {cls_id}",
        )

    ax.set_title(
        f"Паттерн: {pattern.name}\n"
        f"Классификатор: {result.classifier_name}\n"
        f"Точность: {result.accuracy * 100:.1f}%",
        fontsize=9,
    )
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_accuracy_table(
    results: list[TrainResult],
    out_path: Path,
) -> None:
    """Сохраняет сводную таблицу точностей в виде PNG.

    Args:
        results: список всех результатов.
        out_path: путь для сохранения .png.
    """
    # Собираем уникальные паттерны и классификаторы
    patterns_order: list[str] = []
    clfs_order: list[str] = []
    for r in results:
        if r.pattern_name not in patterns_order:
            patterns_order.append(r.pattern_name)
        if r.classifier_name not in clfs_order:
            clfs_order.append(r.classifier_name)

    acc: dict[tuple[str, str], float] = {
        (r.pattern_name, r.classifier_name): r.accuracy for r in results
    }

    table_data = [
        [f"{acc.get((p, c), float('nan')) * 100:.1f}%" for p in patterns_order]
        for c in clfs_order
    ]

    fig, ax = plt.subplots(
        figsize=(max(8, len(patterns_order) * 3 + 3), len(clfs_order) * 0.5 + 2)
    )
    ax.axis("off")
    tbl = ax.table(
        cellText=table_data,
        rowLabels=clfs_order,
        colLabels=patterns_order,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    ax.set_title("Точность классификаторов по паттернам", fontsize=11, pad=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def print_results_table(results: list[TrainResult]) -> None:
    """Выводит сводную таблицу точностей в консоль.

    Args:
        results: список всех результатов.
    """
    patterns_order: list[str] = []
    clfs_order: list[str] = []
    for r in results:
        if r.pattern_name not in patterns_order:
            patterns_order.append(r.pattern_name)
        if r.classifier_name not in clfs_order:
            clfs_order.append(r.classifier_name)

    acc: dict[tuple[str, str], float] = {
        (r.pattern_name, r.classifier_name): r.accuracy for r in results
    }

    col_w = 30
    clf_w = 28

    header = " " * clf_w + "".join(p[:col_w].ljust(col_w) for p in patterns_order)
    print(header)
    print("-" * len(header))
    for c in clfs_order:
        row = c[:clf_w].ljust(clf_w)
        for p in patterns_order:
            val = acc.get((p, c), float("nan"))
            row += f"{val * 100:.1f}%".ljust(col_w)
        print(row)
