# 📄 (다빈님 전용) 포트폴리오/면접 가이드

이 프로젝트는 단순한 '장난감 프로젝트'가 아니라, **"실제 기업의 KPI(Lift@K)를 해결하기 위한 MLOps 프로젝트"**입니다.
면접이나 자소서에서 **다빈님(DL 담당)**이 강조해야 할 핵심 포인트를 정리해드립니다.

---

## 🏗️ 1. 담당 역할 (My Contribution)
> "저는 이 프로젝트에서 **Deep Learning(MLP) 파이프라인 구축 및 최적화**를 전담했습니다."

- **핵심 키워드**: `PyTorch`, `Deep Ensemble`, `Optuna`, `MLOps (Artifacts Management)`
- **구체적인 기여**:
    1. **Baseline 구축**: 초기 MLP 모델(Base)부터 심화 모델(Advanced)까지 단계적 고도화.
    2. **최적화(Optimization)**: Optuna를 활용해 **학습률(Learning Rate)과 은닉층(Hidden Dim)** 최적값 탐색.
    3. **안정성(Stability)**: 앙상블(Ensemble, 5개 모델 평균) 기법을 도입해, 단일 모델의 불안정한 예측을 보정하고 일반화 성능 확보.

---

## 🚨 2. 문제 해결 (Troubleshooting) - 면접 필살기

**Q. 딥러닝(DL)이 머신러닝(ML/Tree)보다 성능이 잘 안 나오지 않았나요?**
> **A. (답변 가이드)**
> "맞습니다. 정형 데이터(Tabular Data) 특성상 Tree 모델이 강력했습니다.
> 하지만 저는 **DL 모델의 가능성**을 끝까지 파고들었습니다.
> 초기에는 Loss가 튀는(Oscillation) 문제가 있었지만, **Batch Normalization**과 **Dropout**을 적절히 섞어 학습을 안정화시켰습니다.
> 결국 앙상블을 통해 ML 모델과 대등한 수준까지 성능을 끌어올렸고, **'다양성(Diversity)'** 측면에서 최종 예측에 기여하는 파이프라인을 완성했습니다."

---

## 🎨 3. 포트폴리오 기술서 예시

### [Project] E-commerce 고객 이탈 예측 (DL 파트 리드)
- **성과**: 상위 5% 고객 타겟팅 효율(Lift) **1.12배** 달성 (Random 대비 구체적 수치)
- **Tech Stack**: Python, PyTorch, Scikit-learn, Optuna
- **Key Action**:
    - **Modular Pipeline**: 실험의 재현성을 위해 모델 정의(`model_definitions.py`)와 학습 코드를 분리하고, 모든 실험 결과(Metric, Weight)가 자동 저장되는 파이프라인 구축.
    - **Deep Ensemble**: 5개의 서로 다른 초기값을 가진 모델을 학습시켜 Soft Voting 하는 앙상블 전략으로 예측 분산 최소화.

---

**💡 팁:**
`notebooks/dl` 폴더에 있는 `MLP_base`, `MLP_enhance`, `MLP_advanced` 이 3단계 과정을 보여주는 것만으로도, **"체계적으로 실험하는 개발자"**라는 인상을 줄 수 있습니다.
