# 유틸리티 모듈 문서 (Utility Modules Documentation)

이 디렉토리는 SKN23-2nd-3Team 프로젝트를 위한 유틸리티 모듈들을 포함하고 있습니다. 파일 경로 관리, 지표 계산, 모델 및 결과물(Artifacts) 저장/불러오기, 시각화 기능을 제공합니다.

## 모듈 개요 (Modules Overview)

### 1. `paths.py`
**목적 (Purpose)**: 프로젝트 전체에서 사용되는 파일 경로를 중앙에서 관리하여 일관성을 유지합니다.
**사용법 (Usage)**:
```python
from app.utils.paths import PATHS

print(PATHS.MODELS_DL)      # 딥러닝 모델 저장 경로 출력
print(PATHS.PREPROCESSING)  # 스케일러 저장 경로 출력
print(PATHS.METRICS)        # 지표 JSON 저장 경로 출력
```

### 2. `metrics.py`
**목적 (Purpose)**: 모델 예측 성능을 평가합니다. 이탈 예측(Churn Prediction)에 특화된 로직(Top-K Lift 등)을 포함합니다.
**주요 함수 (Key Functions)**:
- `evaluate_churn_metrics(y_true, y_prob)`:
    - **입력**: 실제값(y_true), 예측 확률(y_prob)
    - **출력**: PR-AUC, 상위 5% 정밀도/재현율/Lift, 그리고 상세 랭킹 테이블이 포함된 딕셔너리를 반환합니다.

### 3. `artifacts.py`
**목적 (Purpose)**: 학습된 모델, 스케일러, 지표, 그래프 등을 표준화된 경로에 저장합니다.
**주요 함수 (Key Functions)**:
- `save_model_and_artifacts(...)`:
    - 모델(.pt/.pkl), 스케일러(.pkl), 지표(.json), 시각화 이미지(.png), 리포트(.md)를 `paths.py`에 정의된 각 폴더에 자동으로 저장합니다. 기존 파일이 있다면 삭제하고 덮어씁니다.

### 4. `load_metrics.py`
**목적 (Purpose)**: 저장된 성능 지표 JSON 파일을 간편하게 불러옵니다.
**사용법 (Usage)**:
```python
from app.utils.load_metrics import load_metrics

# 'mlp_enhance' 모델의 성능 지표 불러오기
metrics = load_metrics("mlp_enhance")
print(metrics["PR-AUC (Average Precision)"])
```

### 5. `load_model.py`
**목적 (Purpose)**: 학습된 모델과 전처리 스케일러를 불러옵니다. DL(PyTorch)과 ML(Sklearn) 모델을 모두 지원합니다.
**사용법 (Usage)**:
```python
from app.utils.load_model import load_model, load_scaler

# PyTorch 모델 불러오기
model = load_model("mlp_enhance", model_type="dl")

# Sklearn 모델 불러오기
ml_model = load_model("rf_baseline", model_type="ml")

# 스케일러 불러오기
scaler = load_scaler("mlp_enhance")
```

### 6. `plotting.py`
**목적 (Purpose)**: 분석을 위한 표준화된 시각화 함수를 제공합니다.
**사용법 (Usage)**:
```python
from app.utils.plotting import plot_pr_curve, plot_confusion_matrix
import matplotlib.pyplot as plt

# PR 곡선 그리기
fig = plot_pr_curve(y_true, y_prob, title="PR Curve")
plt.show()

# 혼동 행렬(Confusion Matrix) 그리기
fig_cm = plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix")
plt.show()
```

## 업데이트 내역
- **빈 파일 구현**: `load_metrics.py`, `load_model.py`, `plotting.py`가 구현되어 유틸리티 기능이 강화되었습니다.
- **경로 일관성**: 모든 모듈은 `paths.py`를 참조하여 경로를 가져오므로, 폴더 구조가 바뀌어도 코드 수정 없이 대응 가능합니다.
- **한국어 문서화**: 모든 설명과 사용법을 상세한 한국어로 제공합니다.
