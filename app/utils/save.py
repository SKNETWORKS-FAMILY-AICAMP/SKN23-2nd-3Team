import json
import joblib
import torch
from app.utils.paths import PATHS


def save_model_and_artifacts(
    model,
    model_name: str,
    model_type: str,
    metrics: dict,
    version: str = "baseline",
    scaler=None,
    figures: dict | None = None,
    config: dict | None = None,
):
    """
    모델 학습 결과 산출물을 공통 규칙에 따라 저장한다.

    이 함수는 ML / DL 모델 저장 방식을 하나로 통합하며,
    전달된 산출물만 명시적으로 저장하는 것을 설계 원칙으로 한다.
    암묵적으로 추론하거나 자동 생성하여 저장하는 동작은 하지 않는다.

    저장되는 항목
    ----------------
    1. model
       - model_type이 'ml'인 경우: joblib을 사용하여 .pkl 파일로 저장
       - model_type이 'dl'인 경우: torch state_dict를 .pt 파일로 저장

    2. metrics
       - 항상 JSON 파일로 저장된다
       - 모델 성능 비교 및 리포트, Streamlit 시각화에 사용된다

    3. figures (선택)
       - figures 딕셔너리에 전달된 matplotlib figure 객체만 저장된다
       - 각 key는 파일명 suffix로 사용되며 .png 이미지로 저장된다
       - 예: {"confusion_matrix": fig_cm, "pr_curve": fig_pr}

    4. scaler (선택)
       - StandardScaler, MinMaxScaler 등 전처리 객체
       - 전달된 경우에만 joblib을 사용하여 .pkl 파일로 저장된다

    5. config
       - 모델 설정 및 재현성을 위한 설정 정보
       - config 인자가 None이어도 기본 정보로 JSON 파일이 반드시 생성된다

    저장되지 않는 항목
    -------------------
    - figures 딕셔너리에 포함되지 않은 그림
    - scaler 인자를 전달하지 않았을 경우 scaler 객체
    - 생성되지 않았거나 함수에 전달되지 않은 figure 객체
    - 모델, metrics, config 외의 암묵적인 정보

    요약하면, 이 함수는 "넘긴 것만 저장"하며
    사용자가 명시적으로 전달한 산출물만 파일로 남긴다.
    """

    file_prefix = f"{model_name}_{version}"
    saved = {}

    # 1. 모델 저장
    if model_type == "ml":
        model_path = PATHS["models_ml"] / f"{file_prefix}.pkl"
        joblib.dump(model, model_path)
    elif model_type == "dl":
        model_path = PATHS["models_dl"] / f"{file_prefix}.pt"
        torch.save(model.state_dict(), model_path)
    else:
        raise ValueError("model_type must be 'ml' or 'dl'")

    saved["model"] = str(model_path)

    # 2. scaler 저장 (전달된 경우에만)
    if scaler is not None:
        scaler_path = PATHS["models_preprocessing"] / f"{file_prefix}_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        saved["scaler"] = str(scaler_path)

    # 3. metrics 저장 (항상 저장)
    metrics_path = PATHS["models_metrics"] / f"{file_prefix}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    saved["metrics"] = str(metrics_path)

    # 4. figures 저장 (figures에 전달된 것만)
    if figures:
        for name, fig in figures.items():
            img_path = PATHS["assets_training"] / f"{file_prefix}_{name}.png"
            fig.savefig(img_path, dpi=150, bbox_inches="tight")
            saved[name] = str(img_path)

    # 5. config 저장 (항상 생성)
    config_payload = config or {
        "model_name": model_name,
        "version": version,
        "feature_source": "features_ml_clean.parquet"
    }

    config_path = PATHS["models_configs"] / f"{file_prefix}_config.json"
    with open(config_path, "w") as f:
        json.dump(config_payload, f, indent=4)
    saved["config"] = str(config_path)

    return saved