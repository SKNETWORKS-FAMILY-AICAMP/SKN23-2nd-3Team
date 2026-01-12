import os
import json
import torch
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from app.utils.paths import PATHS


def save_model_and_artifacts(
    model,
    model_name,
    model_type,
    metrics,
    scaler=None,
    figures=None,
    config=None,
    report=None,
):
    """
    모델과 관련 결과물(Artifacts)을 지정된 경로에 저장합니다.

    Args:
        model: 학습된 모델 객체.
        model_name (str): 모델 파일 이름 (예: 'mlp_enhance').
        model_type (str): 'dl' (PyTorch) 또는 'ml' (Sklearn).
        metrics (dict): evaluate_churn_metrics의 결과 딕셔너리.
        scaler (object, optional): 스케일러 객체.
        figures (dict, optional): {'이름': figure} 형태의 시각화 그래프 딕셔너리.
        config (dict, optional): 저장할 하이퍼파라미터 설정값.
        report (str, optional): Markdown 리포트 내용.

    사용 예시:
        >>> save_model_and_artifacts(
                model=my_model,
                model_name="mlp_demo",
                model_type="dl",
                metrics=my_metrics,
                scaler=my_scaler,
                figures={"pr_curve": fig}
            )
    """

    print(f">>> [Artifacts] Starting saving process for {model_name}...")

    # Helper for clean saving
    def clean_save_path(path):
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"  - [Clean] Deleted existing file: {path}")
            except OSError as e:
                print(f"  - [Warning] Could not delete {path}: {e}")

    # Determine base path for models/scalers
    if model_type == "dl":
        base_model_path = PATHS.MODELS_DL
    elif model_type == "ml":
        base_model_path = PATHS.MODELS_ML
    else:
        raise ValueError("model_type must be 'dl' or 'ml'")

    # 1. Save Model
    if model_type == "dl":
        model_path = os.path.join(base_model_path, f"{model_name}.pt")
        clean_save_path(model_path)
        torch.save(model.state_dict(), model_path)
    elif model_type == "ml":
        model_path = os.path.join(base_model_path, f"{model_name}.pkl")
        clean_save_path(model_path)
        joblib.dump(model, model_path)

    print(f"  - Model saved to {model_path}")

    # 2. Save Metrics
    metrics_path = os.path.join(PATHS.METRICS, f"{model_name}_metrics.json")
    clean_save_path(metrics_path)

    # Combine metrics and config if provided
    final_metrics = metrics.copy()
    if config:
        final_metrics["config"] = config

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=4)

    print(f"  - Metrics saved to {metrics_path}")

    # 3. Save Scaler
    if scaler:
        scaler_path = os.path.join(base_model_path, f"{model_name}_scaler.pkl")
        clean_save_path(scaler_path)
        joblib.dump(scaler, scaler_path)
        print(f"  - Scaler saved to {scaler_path}")

    # 4. Save Figures
    if figures:
        for fig_name, fig in figures.items():
            fig_path = os.path.join(
                PATHS.ASSETS_TRAINING, f"{model_name}_{fig_name}.png"
            )
            clean_save_path(fig_path)
            fig.savefig(fig_path, bbox_inches="tight")
            print(f"  - Figure '{fig_name}' saved to {fig_path}")

    # 5. Save Report
    if report:
        report_path = os.path.join(PATHS.REPORTS_TRAINING, f"{model_name}_report.md")
        clean_save_path(report_path)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  - Report saved to {report_path}")

    print(f">>> [Artifacts] All artifacts saved successfully.")
