from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np


def evaluate_churn_metrics(y_true, y_prob):
    """
    이탈 예측(Churn Prediction) 모델의 성능을 평가합니다.
    PR-AUC와 Top-K%에 대한 Lift, Precision, Recall을 계산합니다.

    Args:
        y_true (array-like): 실제값 (0 또는 1).
        y_prob (array-like): 이탈할 확률 (예측값, 0~1 사이).

    Returns:
        dict: 요약 지표와 상세 랭킹 테이블이 포함된 딕셔너리.

    사용 예시:
        >>> metrics = evaluate_churn_metrics(y_test, y_prob)
        >>> print(metrics['상위 5% 리프트 (Lift)'])
    """

    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # 1. PR-AUC (Average Precision)
    pr_auc = average_precision_score(y_true, y_prob)

    # 2. Ranking Performance (Focus on Top 5%)
    base_rate = y_true.mean()
    df_res = pd.DataFrame({"label": y_true, "prob": y_prob})
    df_res = df_res.sort_values(by="prob", ascending=False)

    # Calculate for Top 5%
    k = 5
    count = int(len(df_res) * (k / 100))
    if count == 0:
        count = 1

    top_k_subset = df_res.iloc[:count]

    # Precision@5%
    top_5_precision = top_k_subset["label"].mean()

    # Recall@5% (Capture Rate)
    total_positives = df_res["label"].sum()
    captured_positives = top_k_subset["label"].sum()
    top_5_recall = captured_positives / total_positives if total_positives > 0 else 0.0

    # Lift@5%
    lift_5 = top_5_precision / base_rate if base_rate > 0 else 0.0

    # 3. Detailed Ranking Table (Lift by Decile/Percentile)
    ranking_list = []
    for k in [5, 10, 15, 20, 25, 30]:
        count_k = int(len(df_res) * (k / 100))
        if count_k == 0:
            count_k = 1

        subset_k = df_res.iloc[:count_k]
        prec_k = subset_k["label"].mean()

        # Captured Positives
        captured_k = subset_k["label"].sum()
        recall_k = captured_k / total_positives if total_positives > 0 else 0.0

        lift_k = prec_k / base_rate if base_rate > 0 else 0.0

        ranking_list.append(
            {
                "Top_K": f"{k}%",
                "Precision": float(prec_k),
                "Recall": float(recall_k),
                "Lift": float(lift_k),
            }
        )

    # Summary Dictionary with Korean Keys
    metrics_result = {
        "PR-AUC (Average Precision)": float(pr_auc),
        "상위 5% 정밀도 (Precision)": float(top_5_precision),
        "상위 5% 재현율 (Recall)": float(top_5_recall),
        "상위 5% 리프트 (Lift)": float(lift_5),
        "ranking": ranking_list,
    }

    return metrics_result
