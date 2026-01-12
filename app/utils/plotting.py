import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, confusion_matrix


def plot_pr_curve(y_true, y_prob, title="PR Curve"):
    """
    Precision-Recall (PR) 곡선을 그립니다.

    Args:
        y_true (array-like): 실제값 (Ground truth labels).
        y_prob (array-like): 예측 확률 (Predicted probabilities).
        title (str): 그래프 제목.

    Returns:
        matplotlib.figure.Figure: 생성된 그래프 객체 (Figure).

    사용 예시:
        >>> fig = plot_pr_curve(y_test, y_pred_prob)
        >>> plt.show()
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, marker=".", label="PR Curve")
    ax.set_xlabel("Recall (재현율)")
    ax.set_ylabel("Precision (정밀도)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """
    혼동 행렬 (Confusion Matrix)을 그립니다.

    Args:
        y_true (array-like): 실제값 (Ground truth labels).
        y_pred (array-like): 예측값 (Predicted labels).
        title (str): 그래프 제목.

    Returns:
        matplotlib.figure.Figure: 생성된 그래프 객체 (Figure).

    사용 예시:
        >>> fig = plot_confusion_matrix(y_test, y_pred_label)
        >>> plt.show()
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted (예측값)")
    ax.set_ylabel("Actual (실제값)")
    ax.set_title(title)

    return fig
