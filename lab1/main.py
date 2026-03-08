"""Точка входа: программно генерирует паттерны, обучает классификаторы,
сохраняет графики и сводную таблицу."""

from __future__ import annotations

import sys
from pathlib import Path

# Добавляем корень проекта в sys.path, чтобы импорты работали из любого CWD
sys.path.insert(0, str(Path(__file__).parent))

from core.classifiers import get_all_classifiers
from core.trainer import TrainResult, train_and_evaluate
from patterns import Pattern, get_all_patterns
from visualizer import plot_decision_boundary, print_results_table, save_accuracy_table

OUT_DIR = Path(__file__).parent / "output"


def run_experiment(pattern: Pattern, exp_num: int) -> list[TrainResult]:
    """Обучает все классификаторы на одном паттерне и сохраняет графики.

    Args:
        pattern: паттерн с данными.
        exp_num: номер эксперимента (для нумерации файлов).

    Returns:
        Список результатов для всех классификаторов.
    """
    print(f"\n=== Паттерн {exp_num}: {pattern.name} ===")
    print(f"    {pattern.description}")
    print(f"    Точек: {len(pattern.data.points)}, классов: {pattern.data.num_classes}")

    results: list[TrainResult] = []
    clfs = get_all_classifiers()

    for clf in clfs:
        result = train_and_evaluate(clf, pattern.data, pattern.name)
        results.append(result)

        safe_clf = type(clf).__name__  # ASCII-имя класса
        img_path = OUT_DIR / f"exp{exp_num}_{safe_clf}.png"
        plot_decision_boundary(result, pattern, img_path)

        print(f"    [{clf.name:30s}]  точность = {result.accuracy * 100:5.1f}%  -> {img_path.name}")

    return results


def main() -> None:
    """Основная функция: запускает все эксперименты и сохраняет итоги."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    patterns = get_all_patterns()

    all_results: list[TrainResult] = []
    for idx, pattern in enumerate(patterns, start=1):
        all_results.extend(run_experiment(pattern, idx))

    print("\n\n=== Сводная таблица точностей ===")
    print_results_table(all_results)

    table_path = OUT_DIR / "accuracy_table.png"
    save_accuracy_table(all_results, table_path)
    print(f"\nТаблица сохранена: {table_path}")


if __name__ == "__main__":
    main()
