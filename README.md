# <div align="center"> **🛒 E-commerce Log 기반 이탈 예측 모델링** </div>
---
> **🛒 전자상거래 로그 기반 고객 이탈 예측 및 타겟팅 최적화 모델링**  


## 👥 팀 소개 
> 팀 명 : 서거니와 아이들 시즌2

| ^^  | ^^  | ^^  | ^^  | ^^ |
| :-:  | :-:  | :-:  | :-:  | :-:  |
| **정석원** | **김다빈** | **김지우** | **송주엽** | **신승훈** |
| 팀장/PM/ML | 팀원/DL| 팀원/ML | 팀원/ML | 팀원/Dev  |
| - 문제정의/라벨/스플릿 설계  <br> - 공통 평가 지표/리더보드 운영 <br>  ML 담당 <br> 발표| DL 파이프라인/학습 코드 <br> | 내용 | 내용 | 내용 |
| <a href="https://github.com/jsrop07"><img src="https://img.shields.io/badge/GitHub-jsrop07-pink?logo=github"></a> | <a href="https://github.com/tree0317"><img src="https://img.shields.io/badge/GitHub-tree0317-red?logo=github"></a> | <a href="https://github.com/jooooww"><img src="https://img.shields.io/badge/GitHub-jooooww-blue?logo=github"></a> | <a href="https://github.com/JUYEOP024"><img src="https://img.shields.io/badge/GitHub-JUYEOP024-black?logo=github"></a> | <a href="https://github.com/seunghun92-lab"><img src="https://img.shields.io/badge/GitHub-seunghun92--lab-white?logo=github"></a> | 



## 📄 프로젝트 개요 (Overview)

### ✦ 프로젝트명
> 🛒 전자상거래 로그 기반 고객 이탈 예측 및 타겟팅 최적화 모델링

### ✦ 프로젝트 기간
> 2026.01.14(수) ~ 2026.01.16(금)


### ✦ 프로젝트 소개
본 프로젝트는 약 378만 건의 전자상거래 이벤트 로그를 기반으로,
특정 시점(Anchor Time)을 기준으로 유저 행동 패턴을 학습하고,
향후 30일 이내 휴면(m2) 전환 가능성을 예측하는 모델을 구축합니다.

또한 본 프로젝트는 단순 모델 성능 경쟁이 아니라
 - 시점 기반 데이터 누수(Data Leakage) 차단
 - Time Split 기반의 실무형 검증 시나리오
 - Top-K(예: 상위 5%) 타겟팅을 전제로 한 평가 체계

를 파이프라인으로 구현하는 데 초점을 둡니다.

### ✦ 프로젝트 필요성 (배경)

**1. 신뢰 이슈가 “리텐션”에 직접 영향**  
국내 이커머스는 가격/배송 경쟁뿐 아니라, 보안·CS·고객경험 같은 ‘신뢰’ 요인이 이용자 유지에 직접적인 영향을 줍니다.
실제로 2025년 11월 말, 쿠팡의 대규모 개인정보 유출(33.7M 계정 규모로 보도) 이슈 이후 이용자 이탈을 뜻하는 ‘탈팡’ 담론이 확산되었고, DAU 변동 및 규제·조사 이슈 등 시장 리스크가 연이어 보도되었습니다

**2. Rule-based 마케팅의 한계**  
“최근 n일 미접속” 같은 규칙은 이미 늦은 사후 대응이 되기 쉽습니다.
로그 기반 행동 패턴에서 이탈 징후를 사전에 포착하여 선제 대응할 수 있는 모델이 필요합니다.

**3. 데이터 기반의 핀셋 타겟팅**  
현실적으로 모든 고객에게 쿠폰/혜택을 제공할 수 없습니다.
따라서 우리는 휴면 확률이 높은 상위 K% 고객을 선별하여 쿠폰을 제공함으로써, 마케팅 비용을 효율적으로 사용하고 리텐션 효과를 극대화하는 운영 시나리오를 가정합니다.

## 💼 비즈니스 이해 (Business Understanding)

### ✦ 이탈(휴면) 정의(라벨)
- **m2(휴면)**: *(예시)* Anchor 시점(t) 이후 **H=30일 동안 구매/접속/핵심 이벤트가 0회**이면 휴면으로 정의한다.  


### ✦ 타겟팅 가정 (Operational Targeting)
- 운영 제약: 마케팅 예산/인력으로 인해 **상위 K% 고위험 고객만 케어 가능**

<!-- ### ✦ 비용/효익 가정 (Cost–Benefit Assumption)
- **이탈(휴면) 1건 손실 = L**, 캠페인(쿠폰) 1건 비용 = C  
- 목표는 “모든 고객 정확 분류”가 아니라, **C를 최소로 쓰면서 L을 최대한 방어하는 Top-K 타겟팅 최적화**이다. -->

## 📏 평가 지표 및 선택 이유 (Metrics & Rationale)

본 프로젝트의 운영 목표는 **휴면상위 K% 고객에게만 쿠폰/케어를 수행**하는 것이다.  
따라서 임계값(0.5) 기반 지표보다 **Top-K 타겟팅 성과**를 직접 측정하는 지표를 우선한다.

### ✦ Main KPI (Primary)
- **Lift@K**: 랜덤 타겟팅 대비 **몇 배 효율적인지**를 직관적으로 보여주는 실무 지표  
  - Lift@K = Precision@K / Prevalence(전체 휴면 비율)

### ✦ Secondary KPI
- **Precision@K**: 상위 K% 타겟의 “순도” (쿠폰 낭비 최소화)
- **Recall@K**: 전체 휴면 중 상위 K%가 커버한 비율 (커버리지)

### ✦ Diagnostic / 참고 지표
- **PR-AUC(AP)**: threshold를 변화시키며 Precision–Recall trade-off를 요약한 지표로,  
  모델의 전반적인 랭킹 품질을 점검하는 용도로 활용한다.

> Note: 본 프로젝트는 “Top-K 운영”이므로, 모델 선택은 Lift@K/Precision@K를 우선하고 PR-AUC는 보조로 사용한다.

## 📄   프로젝트 목표

**1. 문제의 실무적 재정의 (Practical Problem Definition)**
복잡한 다중 분류 대신, 실제 마케팅 액션(쿠폰 발송 등)이 가능한 **휴면(m2) 여부 이진 분류**로 문제를 단순화하여 실용성을 높였습니다.

**2. 비즈니스 중심의 평가지표 (Business-Aligned Metrics)**  
운영 시나리오가 **“상위 K%만 타겟”**이므로, 임계값(예: 0.5) 기반 지표만으로는 부족합니다.  
따라서 랭킹 기반 지표를 사용합니다.  
- PR-AUC(AP): PR 곡선의 요약(Threshold-free)
- Precision@TopK(%) / Recall@TopK(%): 상위 K%의 타겟팅 품질
- Lift@TopK(%): 랜덤 대비 효율(타겟팅 모델의 실무 지표)

**3. 재현 가능한 파이프라인 (Reproducible Pipeline)**
- 모든 모델이 동일한 Anchor/Label/Split을 공유하도록 표준화
- Time Split로 Look-ahead(미래 정보) 누수 위험 최소화
- 아티팩트(모델/스케일러/피처 순서/평가 결과)를 표준 디렉토리에 저장

## 📏 평가 지표 및 선택 이유 (Metrics & Rationale)

본 프로젝트의 운영 목표는 **휴면상위 K% 고객에게만 쿠폰/케어를 수행**하는 것이다.  
따라서 임계값(0.5) 기반 지표보다 **Top-K 타겟팅 성과**를 직접 측정하는 지표를 우선한다.

### ✦ Main KPI (Primary)
- **Lift@K**: 랜덤 타겟팅 대비 **몇 배 효율적인지**를 직관적으로 보여주는 실무 지표  
  - Lift@K = Precision@K / Prevalence(전체 휴면 비율)

### ✦ Secondary KPI
- **Precision@K**: 상위 K% 타겟의 “순도” (쿠폰 낭비 최소화)
- **Recall@K**: 전체 휴면 중 상위 K%가 커버한 비율 (커버리지)

### ✦ Diagnostic / 참고 지표
- **PR-AUC(AP)**: threshold를 변화시키며 Precision–Recall trade-off를 요약한 지표로,  
  모델의 전반적인 랭킹 품질을 점검하는 용도로 활용한다.

> Note: 본 프로젝트는 “Top-K 운영”이므로, 모델 선택은 Lift@K/Precision@K를 우선하고 PR-AUC는 보조로 사용한다.
## 📂 프로젝트 설계 
```
📦 SKN23-2nd-3Team/
├── data/
│   ├── raw/
│   │   └── 원본.csv
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


## 📄   프로젝트 내용
### 데이터셋 (Dataset)
**1. 데이터 정의**
- 기간: 7개월 이커머스 로그
- 대상: watch 카테고리
- 브랜드: samsung / apple / xiaomi
- 데이터 형태: 대용량 이벤트/거래 로그 기반의 사용자 단위 예측
- train(11~1월)/val(2월)/test(3월)로 학습/검증/테스트로 나눠서 진행

<table>
  <tr>
    <td align="center" width="50%">
      <figure>
        <figcaption>📌 상세 라벨 분포(m0 : 정상 / m1 : 스위치(브랜드 전환) / m2 : 휴면)</figcaption>
        <img src="assets\eda\label_distribution.png" width="100%" />
      </figure>
    </td>
    <td align="center" width="50%">
      <figure>
        <figcaption>📌 휴면비율(m2/ m0+m1)</figcaption>
                <img src="assets\eda\target_rate.png" width="100%" />
      </figure>
    </td>
  </tr>
</table>

**2 제거/제외 컬럼(또는 후보) & 이유**
-  `home_brand`, `future_brand`  
  → **라벨 이후 정보가 포함될 가능성(미래 정보/사후 정보)**
  → **train/val/test 경계를 침범할 가능성 -> 데이터 누수확률 증가**
<img src="assets\eda\data_leak.png" width="100%" />

**3 집계 기간(윈도우W)**
- 과거 W=30일: **[t-30, t]** 구간만 사용
- 미래 H=30일: **(t, t+30]** 구간으로 라벨 산출
- 원칙: **event_time < anchor_time** 조건을 만족하는 데이터만 피처 생성에 사용

**4 파생변수 목록**
- Recency(경과일): 마지막 활동/구매가 오래될수록 휴면 가능성 증가

- Frequency/Monetary(활동·구매·금액): 최근 활동·구매·지출이 많을수록 휴면 가능성 감소

- Trend/Regularity(추세·규칙성): 최근 활동이 줄고 방문이 불규칙해질수록 휴면 가능성 증가 
  
| 파생변수(변수명)                   | 한글 이름           | 파생변수 설명                                   | 값이 높을수록 휴면확률            | 분류(Recency/Frequency/Monetary/기타) |
| --------------------------- | --------------- | ----------------------------------------- | ----------------------- | --------------------------------- |
| `n_events_30d`              | 최근 30일 활동 수     | Lookback 30일 내 전체 이벤트(조회/클릭/장바구니/구매 등) 횟수 | **감소(↓)**               | **Frequency**                     |
| `active_days_30d`           | 최근 30일 활동 일수    | Lookback 30일 중 이벤트가 발생한 서로 다른 날짜 수        | **감소(↓)**               | **Frequency**                     |
| `n_purchase_30d`            | 최근 30일 구매 횟수    | Lookback 30일 구매 완료 건수                     | **감소(↓)**               | **Monetary / Frequency**          |
| `purchase_ratio`            | 구매 전환율          | 구매 관련 이벤트 비율(예: 구매수/전체 이벤트수 또는 구매/조회 등)   | **감소(↓)**               | **Engagement(기타)**                |
| `days_since_last_event`     | 마지막 활동 경과일      | Anchor 기준 마지막 이벤트 이후 경과일                  | **증가(↑)**               | **Recency**                       |
| `days_since_last_purchase`  | 마지막 구매 경과일      | Anchor 기준 마지막 구매 이후 경과일                   | **증가(↑)**               | **Recency**                       |
| `brand_concentration_ratio` | 브랜드 집중도         | 특정 브랜드(또는 상위1개 브랜드) 활동/구매 비중(집중도)         | **상황 의존(±)**            | **Loyalty(기타)**                   |
| `brand_switch_count_30d`    | 최근 30일 브랜드 전환 수 | Lookback 30일 동안 브랜드가 바뀐 횟수(연속 구매/조회 기준)   | **증가(↑)**               | **Loyalty(기타)**                   |
| `total_spend_30d`           | 최근 30일 총 구매 금액  | Lookback 30일 결제 금액 합                      | **감소(↓)**               | **Monetary**                      |
| `activity_ratio_15d`        | 최근 15일 활동 비율    | 최근 15일 활동량 / 30일 활동량 등 “최근성 가중” 비율        | **감소(↓)**               | **Recency / Trend(기타)**           |
| `price_volatility`          | 가격 민감도(변동성)     | 사용자가 반응한 가격의 변동폭/편차(또는 할인 반응성)            | **증가(↑)**               | **Price(기타)**                     |
| `n_events_7d`               | 최근 7일 활동 수      | Lookback 7일 내 이벤트 수(단기 참여도)               | **감소(↓)**               | **Frequency**                     |
| `visit_regularity`          | 방문 규칙성          | 방문 간격의 규칙성(예: 간격 분산/표준편차 기반)              | **감소(↓)**               | **Engagement(기타)**                |
| `activity_trend`            | 활동 추세           | 최근 구간 대비 활동 증가/감소 추세(예: 15일 vs 30일, 기울기)  | **감소(↓)** *(감소 추세일수록↑)* | **Trend(기타)**                     |

**5 파생변수 상관관계 및 다중공선성**
<table>
  <tr>
    <td align="center" width="50%">
      <figure>
        <figcaption>📌 파생변수 상관관계 </figcaption>
        <img src="assets\eda\feature_corr_heatmap.png" width="100%" />
      </figure>
    </td>
    <td align="center" width="50%">
      <figure>
        <figcaption>📌 다중공선성(5이상시 위험) </figcaption>
                <img src="assets\eda\vif.png" width="100%" />
      </figure>
    </td>
  </tr>
</table>


**5  누수 방지 규칙(필수)**
- 스케일러/인코더/결측치 대체 등 전처리는 **train에서만 fit**, val/test는 transform만 수행
- 모델 선택/튜닝은 validation까지만 사용하고, test는 최종 1회 평가

### 🧠 모델링 스토리라인 (Baseline → Strong ML → DL -> Strong DL)
1) **Baseline (M1)**  
- 단순/해석 가능한 모델 Logistic Regression로 기준선 확보

2) **Strong ML (M2~M3)**  
- LightGBM / HGB로 성능 상향 및 안정화

3) **Deep Learning (M4~M5)**  
- MLP를 기본버전부터 성능을 확장시킨 버전 마지막으로 최강의 버전?

모든 실험은 동일한 Anchor/Label/Time Split을 공유하며,  
평가 방식(PR-AUC, Precision/Recall/Lift@K)도 동일하게 유지하여 공정 비교한다.

---

### 모델별 성능 비교 
- 필요 자료 -> 모델 평가지표 비교 그래프 및 표

### 대표 모델 분석
- 필요 자료 -> 모델명 / 선정 이유 / 시각화

### 대표 코드


### 

### 시연 영상

---

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
전체 유저 중 휴면(m2) 유저 비율이 81% 이상으로 압도적으로 높아,  
모든 유저를 휴면으로 예측해도 Accuracy가 90% 이상 나오는 왜곡이 발생했습니다.

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