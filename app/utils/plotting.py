from __future__ import annotations

import platform
from typing import Optional, Sequence

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from sklearn.metrics import precision_recall_curve, confusion_matrix


def configure_matplotlib_korean() -> Optional[str]:
    system = platform.system()

    if system == "Darwin":
        candidates = ["AppleGothic", "Pretendard", "Noto Sans CJK KR", "NanumGothic", "DejaVu Sans"]
    elif system == "Windows":
        candidates = ["Malgun Gothic", "Pretendard", "Noto Sans CJK KR", "NanumGothic", "DejaVu Sans"]
    else:
        candidates = ["NanumGothic", "Noto Sans CJK KR", "Pretendard", "DejaVu Sans"]

    chosen = None
    for name in candidates:
        try:
            fm.findfont(name, fallback_to_default=False)
            chosen = name
            break
        except Exception:
            continue

    if chosen:
        mpl.rcParams["font.family"] = chosen
    mpl.rcParams["axes.unicode_minus"] = False
    return chosen


def plot_pr_curve(y_true, y_prob, title: str = "PR Curve", pr_auc: float | None = None):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2)

    ax.set_xlabel("Recall (재현율)")
    ax.set_ylabel("Precision (정밀도)")
    ax.set_title(title)

    if pr_auc is not None:
        ax.legend([f"PR-AUC = {float(pr_auc):.5f}"])
    else:
        ax.legend(["PR Curve"])

    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true,
    y_pred,
    title: str = "Confusion Matrix",
    labels: Sequence[str] = ("비이탈(m1)", "이탈(m2)"),
    cmap: str = "Blues",
):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap=cmap, interpolation="nearest", aspect="equal")
    fig.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted (예측값)")
    ax.set_ylabel("Actual (실제값)")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    ax.set_xlim(-0.5, cm.shape[1] - 0.5)
    ax.set_ylim(cm.shape[0] - 0.5, -0.5)

    fig.tight_layout()
    return fig