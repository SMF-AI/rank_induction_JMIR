from typing import List, Tuple

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import (
    auc,
    roc_curve,
    confusion_matrix,
    precision_recall_curve,
)


def plot_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[plt.Figure, plt.Axes]:
    """그림을 사용하여 AUROC (Area Under the Receiver Operating Characteristic Curve)를 시각화

    Args:
        y_true (np.ndarray): 실제 라벨 값의 배열.
        y_prob (np.ndarray): 모델의 예측 확률 값의 배열.

    Returns:
        Tuple[plt.Figure, plt.Axes]: Matplotlib의 Figure와 Axes 객체를 반환

    Example:
        >>> fig, axes = plot_auroc(bag_labels, bag_probs)
        >>> plt.savefig("auroc.png")
        >>> mlflow.log_artifact("auroc.png")
        >>> os.remove("auroc.png")
        >>> plt.clf()
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, axes = plt.subplots()
    axes.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUROC = {roc_auc:.3f}")
    axes.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axes.set_xlim([0.0, 1.0])
    axes.set_ylim([0.0, 1.05])
    axes.set_xlabel("False Positive Rate")
    axes.set_ylabel("True Positive Rate")
    axes.set_title("Receiver Operating Characteristic (ROC)")
    axes.legend(loc="lower right")

    return fig, axes


def plot_prauc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[plt.Figure, plt.Axes]:
    """그림을 사용하여 AUROC (Area Under the Receiver Operating Characteristic Curve)를 시각화

    Args:
        y_true (np.ndarray): 실제 라벨 값의 배열.
        y_prob (np.ndarray): 모델의 예측 확률 값의 배열.

    Returns:
        Tuple[plt.Figure, plt.Axes]: Matplotlib의 Figure와 Axes 객체를 반환

    Example:
        >>> fig, axes = plot_auroc(bag_labels, bag_probs)
        >>> plt.savefig("auroc.png")
        >>> mlflow.log_artifact("auroc.png")
        >>> os.remove("auroc.png")
        >>> plt.clf()
    """
    precision, recall, threshold = precision_recall_curve(y_true, y_prob)
    roc_auc = auc(recall, precision)

    fig, axes = plt.subplots()
    axes.plot(
        recall, precision, color="darkorange", lw=2, label=f"PRAUC = {roc_auc:.3f}"
    )
    axes.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    axes.set_xlim([0.0, 1.0])
    axes.set_ylim([0.0, 1.05])
    axes.set_xlabel("Recall")
    axes.set_ylabel("Precision")
    axes.set_title("Receiver Operating Characteristic (PRAUC)")
    axes.legend(loc="lower right")

    return fig, axes


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: List[str],
) -> Tuple[plt.Figure, plt.Axes]:
    """Confusion Matrix를 시각화하는 함수

    Args:
        y_true (np.ndarray): 실제 라벨 값의 배열.
        y_pred (np.ndarray): 모델의 예측 라벨 값의 배열.
        labels (List[str]): 클래스 라벨의 리스트(Ascending order).

    Returns:
        Tuple[plt.Figure, plt.Axes]: Matplotlib의 Figure와 Axes 객체를 반환

    Example:
        >>> fig, axes = plot_confusion_matrix(true_labels, predicted_labels, ["Class 0", "Class 1"])
        >>> plt.savefig("confusion_matrix.png")
        >>> mlflow.log_artifact("confusion_matrix.png")
        >>> os.remove("confusion_matrix.png")
        >>> plt.clf()
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes,
    )

    axes.set_xlabel("Predicted labels")
    axes.set_ylabel("True labels")
    axes.set_title("Confusion Matrix")

    return fig, axes
