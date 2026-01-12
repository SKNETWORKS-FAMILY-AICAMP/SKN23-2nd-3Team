from pathlib import Path

# 프로젝트 루트 디렉토리
# app/utils/paths.py 기준으로 저장소 최상위 디렉토리
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 프로젝트 전역에서 사용하는 공통 경로 정의
PATHS = {
    # 모델
    "models_ml": PROJECT_ROOT / "models/ml",
    "models_dl": PROJECT_ROOT / "models/dl",

    # 전처리 / 재현성
    "models_preprocessing": PROJECT_ROOT / "models/preprocessing",
    "models_configs": PROJECT_ROOT / "models/configs",

    # 평가 결과
    "models_metrics": PROJECT_ROOT / "models/metrics",

    # 시각화 / 리포트
    "assets_training": PROJECT_ROOT / "assets/training",
    "reports_training": PROJECT_ROOT / "reports/training",
}

# 정의된 경로가 없을 경우 자동 생성
for p in PATHS.values():
    p.mkdir(parents=True, exist_ok=True)