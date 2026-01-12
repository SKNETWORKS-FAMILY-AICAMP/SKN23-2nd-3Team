# <div align="center"> **🛒 E-commerce Log 기반 이탈 예측 모델링** </div>
---
> **🛒 전자상거래 로그 기반 고객 이탈 예측 및 타겟팅 최적화 모델링**  


## 👥 팀 소개 
> 팀 명 : 서거니와 아이들 시즌2

| ^^  | ^^  | ^^  | ^^  | ^^ |
| :-:  | :-:  | :-:  | :-:  | :-:  |
| **정석원** | **김다빈** | **김지우** | **송주엽** | **신승훈** |
| 팀장/PM/ML | 팀원/Dev| 팀원/ML | 팀원/ML | 팀원/Dev  |
| 내용 | 내용 | 내용 | 내용 | 내용 |
| <a href="https://github.com/jsrop07"><img src="https://img.shields.io/badge/GitHub-jsrop07-pink?logo=github"></a> | <a href="https://github.com/tree0317"><img src="https://img.shields.io/badge/GitHub-tree0317-red?logo=github"></a> | <a href="https://github.com/jooooww"><img src="https://img.shields.io/badge/GitHub-jooooww-blue?logo=github"></a> | <a href="https://github.com/JUYEOP024"><img src="https://img.shields.io/badge/GitHub-JUYEOP024-black?logo=github"></a> | <a href="https://github.com/seunghun92-lab"><img src="https://img.shields.io/badge/GitHub-seunghun92--lab-white?logo=github"></a> | 



## 📄 프로젝트 개요 (Overview)

### ✦ 프로젝트명
> 🛒 전자상거래 로그 기반 고객 상태 예측 및 타겟팅 최적화 모델링

### ✦ 프로젝트 기간


> 2026.01.14(수) ~ 2026.01.16(금)


### ✦ 프로젝트 소개
본 프로젝트는 약 **378만 건**의 전자상거래 이벤트 로그를 기반으로,  
특정 시점(Anchor Time) 기준 유저 행동 패턴을 학습하고  
향후 30일 이내 휴면 여부(m2)를 예측하는 모델을 구축했으며,

이는 단순 이탈 예측이 아니라
피처 생성 단계부터 데이터 누수(Data Leakage)를 철저히 차단한  
실무 환경을 가정한 파이프라인을 구현하는 데 초점을 두었습니다.

( 최종 결과물은 실제 마케팅 캠페인(쿠폰 발송 등)에 바로 적용 가능한  
타겟팅 기준으로 활용 가능하도록 설계했습니다. )  ---> 💟 요거는 보류


### ✦ 프로젝트 필요성 (배경)

**1. 구독 피로도와 경쟁 심화 (High CAC vs Low Retention)**
이커머스 시장의 포화로 인해 고객의 브랜드 전환(Switching)이 잦아지고 있습니다. 신규 고객 유치 비용(CAC)은 기존 고객 유지 비용보다 **약 5~25배** 더 비싸기 때문에, 기존 고객의 이탈을 막는 것이 수익성에 직결됩니다.

**2. Rule-based 마케팅의 한계**
"최근 3달간 접속 안 함"과 같은 사후약방문식 규칙은 이미 마음이 떠난 고객을 되돌리기 어렵습니다. 이탈 징후를 **사전에 포착**하여 선제적으로 대응할 수 있는 AI 모델이 필요합니다.

**3. 데이터 기반의 핀셋 타겟팅**
직관이 아닌 유저의 행동 로그(클릭, 장바구니 등)를 분석하여 **'이탈 확률 상위 20% 고객'**을 선별하고, 마케팅 예산을 집중하여 ROI를 극대화해야 합니다.








### ✦ 프로젝트 목표

**1. 문제의 실무적 재정의 (Practical Problem Definition)**
복잡한 다중 분류 대신, 실제 마케팅 액션(쿠폰 발송 등)이 가능한 **휴면(Dormant, m2) 여부 이진 분류**로 문제를 단순화하여 실용성을 높였습니다.

**2. 비즈니스 중심의 평가지표 (Business-Aligned Metrics)**
클래스 불균형 상황에서 **단순 정확도(Accuracy)의 함정**을 피하기 위해 **PR-AUC**를 메인 지표로 삼고, 실제 타겟팅 효율을 보기 위해 **Precision@TopK**를 보조 지표로 활용했습니다.

**3. 재현 가능한 파이프라인 (Reproducible Pipeline)**
팀원 간 협업을 위해 모든 모델이 동일한 `Anchor Time`과 `Label`을 공유하도록 설계하여, 모델 간 성능을 공정하게 비교할 수 있는 환경을 구축했습니다.


## 📂 프로젝트 설계 
```
📦 SKN23-2nd-3Team/
├── data/
│   ├── raw/
│   │   └── *.parquet
│   └── processed/
│       └── *.parquet
│
├── app/
│   ├── *.py
│   ├── pages/
│   │   └── *.py
│   └── utils/
│       └── *.py
│
├── models/
│   ├── *.py
│   ├── configs/
│   │   └── *.json
│   ├── preprocessing/
│   │   └── *.pkl
│   ├── ml/
│   │   └── *.pkl
│   ├── dl/
│   │   └── *.pt
│   ├── metrics/
│   │   └── *.json
│   └── eval/
│       └── *.json
│
├── reports/
│   ├── preprocessing/
│   │   └── *.json
│   ├── training/
│   │   └── *.md
│   └── insights/
│       └── *.md
│
├── assets/
│   ├── eda/
│   │   └── *.png
│   ├── training/
│   │   └── *.png
│   └── ui/
│       └── 
│
└── notebooks/
    ├── ml/
    │   └── *.ipynb
    └── dl/
        └── *.ipynb

📁 README.md
📁 requirements.txt
📁 .gitignore

```


## 🛠️ 기술 스택

### Backend / ETL
![Python](https://img.shields.io/badge/Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

### Dashboard / Visualization
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
  <img src="https://img.shields.io/badge/html5-E34F26?style=for-the-badge&logo=html5&logoColor=white"> 
  <img src="https://img.shields.io/badge/css-1572B6?style=for-the-badge&logo=css3&logoColor=white"> 

### VCS
<img src="https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white"> <img src="https://img.shields.io/badge/github-181717?style=for-the-badge&logo=github&logoColor=white">
 


## 📊 수행 결과









## 📎 기타 

### 🚀 Troubleshooting
## 🛠️ Project Troubleshooting & Technical Retrospective

본 프로젝트 수행 과정에서 직면한 핵심 기술적 난관과, 이를 해결하기 위해 적용한 엔지니어링적 의사결정 및 개선 사항을 정리했습니다.

---

## 1. 데이터 신뢰성 및 무결성 확보  
### (Data Integrity & Leakage Prevention)

### 🚨 1-1. 시계열 데이터 누수 (Temporal Data Leakage)

**문제 상황**  
초기 모델 학습 시 AUC 및 Precision이 비현실적으로 높게 측정되었습니다.  
실제 서비스 환경에서는 재현 불가능한 성능이라 판단하여 데이터 누수를 의심했습니다.

**원인 분석**  
피처 엔지니어링 과정에서 `n_events_30d` 등의 집계 변수가 기준 시점(`anchor_time`) 이후의 로그까지 포함하여 계산되고 있었습니다.  
즉, 모델이 미래 정보를 미리 참조한 상태로 학습되는 구조적 누수가 발생했습니다.

**해결 방안**

- **Strict Timestamp Filtering**  
  원시 로그(`base.parquet`)와 `anchor_time`을 기준으로  
  `event_time < anchor_time` 조건을 만족하는 과거 데이터만 집계하도록 파이프라인을 재설계했습니다.
- **Composite Key Merge**  
  데이터 병합 시 단일 `user_id`가 아닌 `(user_id, anchor_time)` 복합 키를 사용하여 시점 불일치를 차단했습니다.

**성과**  
모델 성능의 허수를 제거하고, 실제 배포 환경에서도 신뢰 가능한 성능 기준을 확보했습니다.

---

### 📉 1-2. 검증 방식 오류: Look-ahead Bias

**문제 상황**  
Sliding Window 방식으로 샘플을 생성하면서 동일 유저가 여러 시점에 중복 등장했습니다.  
Random Split 사용 시, 동일 유저의 미래 정보가 Train Set에 포함되는 정보 유출이 발생했습니다.

**해결 방안**

- **Out-of-Time (OOT) Validation** 적용  
- Random Split을 배제하고, 과거 기간으로 학습 → 최신 기간으로 검증하는  
  **Time-based Split** 전략을 채택하여 실제 서비스 환경과 동일한 평가 시나리오를 구성했습니다.

---

## 2. 모델링 전략 및 학습 최적화  
### (Modeling Strategy)

### ⚖️ 2-1. 클래스 불균형 문제 (Class Imbalance)

**문제 상황**  
전체 유저 중 휴면(m2) 유저 비율이 5% 미만으로 매우 낮아,  
모든 유저를 비휴면으로 예측해도 Accuracy가 90% 이상 나오는 왜곡이 발생했습니다.

**해결 방안**

- **Problem Redefinition**  
  다중 분류 대신, 마케팅 액션이 명확한  
  `휴면(m2) vs 비휴면` 이진 분류로 문제를 재정의했습니다.
- **Metric Shift**  
  Accuracy 대신 PR-AUC를 메인 지표로 채택했으며,  
  실무 타겟팅 효율을 고려해 Precision@TopK 지표를 추가했습니다.
- **Targeted SMOTE**  
  Train Set에 한해 SMOTE를 적용하고,  
  Test Set은 원본 분포를 유지하여 평가 신뢰성을 확보했습니다.

---

### 🧪 2-2. 학습 불안정성 및 하이퍼파라미터 탐색

**문제 상황**  
MLP 모델 학습 시 Loss 진동(Oscillation)이 발생했고,  
Grid Search 방식의 비효율성으로 최적 모델 탐색에 한계가 있었습니다.

**해결 방안**

- **Stabilization**  
  Batch Normalization과 Dropout을 적용하여  
  Internal Covariate Shift를 완화하고 학습 안정성을 확보했습니다.
- **Bayesian Optimization**  
  Optuna(TPE 알고리즘)를 도입해 탐색 효율을 약 3배 이상 향상시켰으며,  
  Hidden Dim 512, Learning Rate 0.0029 등의 최적 설정을 도출했습니다.

---

## 3. 추론 파이프라인 및 MLOps  
### (Inference & MLOps)

### 🔄 3-1. 학습–추론 불일치 (Training–Inference Skew)

**문제 상황**  
학습 시 적용된 전처리(Scaling)가 추론 단계에서 누락되거나,  
피처 순서 불일치로 인해 예측 값이 왜곡되는 문제가 발생했습니다.

**해결 방안**

- **Feature Contract**  
  `FEATURE_ORDER`를 단일 기준(Single Source of Truth)으로 정의하여  
  학습과 추론 간 입력 컬럼 순서 및 타입을 강제로 일치시켰습니다.
- **Scaler Synchronization**  
  학습 시 적합된 `scaler.pkl`을 아티팩트로 저장하고,  
  추론 시 `transform`만 수행하도록 파이프라인을 통합했습니다.  
  (ML(Tree) 모델과 DL(MLP) 모델의 전처리 정책은 분리 적용)

---

### 🧹 3-2. 데이터 품질 이슈: 빈 문자열 처리

**문제 상황**  
미래 활동이 없는 유저의 경우 결측치가 NaN이 아닌  
빈 문자열("")로 저장된 케이스가 존재했습니다.  
기존 `isna()` 로직이 이를 정상 값으로 인식해 휴면 판별 오류가 발생했습니다.

**해결 방안**

- **Defensive Coding**  
  전처리 단계에서 빈 문자열("")을 강제로 `np.nan`으로 치환했습니다.
- **Direct Labeling**  
  결측 여부 기반 판단 대신  
  `label == "m2"` 값을 직접 참조하도록 로직을 변경해  
  Rule Agreement 100%를 달성했습니다.

---

### 📦 3-3. 재현성 확보 및 코드 구조 개선

**문제 상황**  
실험 노트북이 난립하며  
최적 모델 및 설정 추적이 어려운 상태(“Notebook Hell”)가 발생했습니다.

**해결 방안**

- **Modularization**  
  모델 아키텍처를 `models/model_definitions.py`로 분리하여  
  노트북 의존성을 제거했습니다.
- **Code as Infrastructure**  
  모델 가중치, 메트릭, 스케일러 등 모든 산출물이  
  표준화된 디렉토리 구조에 자동 저장되도록 파이프라인을 정비했습니다.


### ✏️ 한 줄 회고

| **이름** | **한 줄 회고** |
| :-: | :-- |
| 정석원 | TBD |
| 김다빈 | TBD |
| 김지우 | TBD |
| 송주엽 | TBD |
| 신승훈 | TBD |


