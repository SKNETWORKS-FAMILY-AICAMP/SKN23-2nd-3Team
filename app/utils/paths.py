# app/utils/paths.py

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PATHS = {
    # ✅ 데이터
    "data_processed": PROJECT_ROOT / "data/processed",

    # 모델 (학습 산출물)
    "models_ml": PROJECT_ROOT / "models/ml",
    "models_dl": PROJECT_ROOT / "models/dl",

    # 전처리 / 재현성
    "models_preprocessing": PROJECT_ROOT / "models/preprocessing",
    "models_configs": PROJECT_ROOT / "models/configs",

    # 실시간 예측 / 실시간 예측에 사용하는 metrics JSON
    "models_metrics": PROJECT_ROOT / "models/metrics",

    # 1단계 평가 결과 (팀 공통 규칙)
    "models_eval": PROJECT_ROOT / "models/eval",

    # 시각화 / 리포트
    "assets_training": PROJECT_ROOT / "assets/training",
    "reports_training": PROJECT_ROOT / "reports/training",
}

for p in PATHS.values():
    p.mkdir(parents=True, exist_ok=True)