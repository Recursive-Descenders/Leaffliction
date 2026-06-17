"""Evaluate a trained random forest on held-out data."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from classification.train.forest import RandomForest

CONFUSION_MATRIX_FILENAME = "confusion_matrix.png"


@dataclass(frozen=True)
class ClassMetrics:
    label: str
    false_positives: int
    false_negatives: int
    accuracy: float


def evaluate(
    forest: RandomForest,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    dst: Path,
) -> float:
    y_pred = forest.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    matrix = confusion_matrix(y_test, y_pred, labels=forest.classes_)
    class_metrics = _per_class_metrics(matrix, forest.classes_)
    _print_per_class_metrics(class_metrics)

    output_path = dst / CONFUSION_MATRIX_FILENAME
    _save_confusion_matrix(matrix, class_metrics, output_path)

    return accuracy


def _per_class_metrics(
    matrix: np.ndarray,
    labels: np.ndarray,
) -> list[ClassMetrics]:
    metrics: list[ClassMetrics] = []

    for index, label in enumerate(labels):
        true_positives = matrix[index, index]
        false_negatives = int(matrix[index, :].sum() - true_positives)
        false_positives = int(matrix[:, index].sum() - true_positives)
        support = int(matrix[index, :].sum())
        class_accuracy = (
            true_positives / support if support > 0 else 0.0
        )

        metrics.append(
            ClassMetrics(
                label=str(label),
                false_positives=false_positives,
                false_negatives=false_negatives,
                accuracy=class_accuracy,
            )
        )

    return metrics


def _print_per_class_metrics(class_metrics: list[ClassMetrics]) -> None:
    print("\nPer-class metrics:")
    print(
        f"{'Class':<30} {'FP':>6} {'FN':>6} {'Accuracy':>10}"
    )
    print("-" * 56)
    for metrics in class_metrics:
        print(
            f"{metrics.label:<30} "
            f"{metrics.false_positives:>6} "
            f"{metrics.false_negatives:>6} "
            f"{metrics.accuracy * 100:>9.2f}%"
        )


def _save_confusion_matrix(
    matrix: np.ndarray,
    class_metrics: list[ClassMetrics],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [metrics.label for metrics in class_metrics]
    figure_height = max(12.0, 8.0 + len(labels) * 0.2)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(14, figure_height),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axis = axes[0]
    table_axis = axes[1]

    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)

    tick_positions = np.arange(len(labels))
    axis.set_xticks(tick_positions)
    axis.set_yticks(tick_positions)
    axis.set_xticklabels(labels, rotation=45, ha="right")
    axis.set_yticklabels(labels)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Confusion matrix")

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            axis.text(
                col_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    table_axis.axis("off")
    table = table_axis.table(
        cellText=[
            [
                metrics.label,
                metrics.false_positives,
                metrics.false_negatives,
                f"{metrics.accuracy * 100:.2f}%",
            ]
            for metrics in class_metrics
        ],
        colLabels=["Class", "FP", "FN", "Accuracy"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)
    table_axis.set_title(
        "Per-class false positives, false negatives, accuracy"
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved confusion matrix: {output_path}")
    return output_path
