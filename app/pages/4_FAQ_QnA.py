import streamlit as st
# ===== util 파일 불러오기 =======
from utils.ui import apply_base_layout, hide_sidebar, top_nav, QnA_ui

st.set_page_config(layout="wide", page_title="FAQ - REVIVE")

# UI 유틸리티 적용
apply_base_layout()
hide_sidebar()
top_nav()
QnA_ui()

# ===== 간격 조정 =====
st.markdown("""
<style>
    /* 1. 최상단 여백 제거 (네비바가 들어갈 공간 확보) */
    .block-container { 
        padding-top: 0rem !important;
        padding-bottom: 3rem; 
    }
    
    /* 2. [핵심] 타이틀(h1) 강제로 위로 끌어올리기 */
    h1 {
        padding-top: 0rem !important;
        margin-top: -2rem !important; /* 이 값을 조절해서 간격을 맞추세요 (-2rem ~ -4rem 추천) */
    }

    /* 3. 네비게이션 바와 본문 사이의 쓸데없는 간격 제거 */
    div[data-testid="stVerticalBlock"] {
        gap: 0.9rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ======= Hero Section =======
# st.markdown("""
# <div class="faq-hero">
#     <div class="faq-hero-title">Q&A</div>
#     <div class="faq-hero-subtitle">
#         🚀 SKN23 2nd 3TEAM 프로젝트의 모델 선택, 평가 지표 등에 대한 
#         궁금증을 해결해드립니다.
#     </div>
# </div>
# """, unsafe_allow_html=True)

st.markdown("# Q&A")

# ===== 검색 기능 ======
st.markdown('<div class="search-container">', unsafe_allow_html=True)
search_query = st.text_input(
    "🔍 FAQ 검색", 
    placeholder="궁금한 키워드를 검색해보세요 (예: F1-score, LightGBM, Parquet)",
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

# 카테고리 필터 (데이터 구조에 맞춰 업데이트)
categories = ["전체", "기획 및 비즈니스", "데이터 및 평가지표", "모델 선정", "엔지니어링 및 실무"]
selected_category = st.radio(
    "카테고리",
    categories,
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")

# ==============================================================================
# FAQ 데이터 구조화 (요청하신 모든 내용을 통합)
# ==============================================================================
faq_data = {
    "기획 및 비즈니스": [
        {
            "question": "수많은 주제 중 왜 '고객 이탈 예측'을 선정했나요?",
            "answer": """
**A. 기업 경영의 핵심 리스크 관리**

기업 경영에서 가장 큰 리스크는 **'고객이 왜 떠나는지 모른 채 떠나보내는 것'**입니다. 
신규 고객 유치 비용(CAC)은 기존 고객 유지 비용보다 **5~25배** 더 높습니다. 
따라서 데이터 기반의 **'선제적 케어'**를 통해 기업의 수익성을 방어하고, 고객의 서비스 경험을 개선하고자 이 주제를 선정했습니다.
            """,
            "tags": ["기획의도", "ROI", "리텐션"]
        },
        {
            "question": "왜 3단계 분류가 아닌 이진(Binary) 분류를 선택했나요?",
            "answer": """
**A. 의사결정의 명확성 (Actionability)**

실무적인 의사결정의 명확성 때문입니다. 
'주의/경고/위험' 같은 모호한 분류보다, **"지금 마케팅 예산을 집행할 것인가?"**라는 명확한 행동 기준(Action Item)을 제시하여 현업의 실행력을 높이기 위함입니다.
            """,
            "tags": ["이진분류", "의사결정", "마케팅"]
        },
        {
            "question": "이 모델은 실제 서비스에 어떻게 활용될 수 있나요?",
            "answer": """
**핵심 활용 시나리오:**

1. **🎯 타겟 마케팅:** 이탈 가능성이 높은 상위 유저를 사전에 식별하여 개인화된 쿠폰 제공
2. **📧 자동화된 리텐션 캠페인:** 이탈 위험 감지 시, 적절한 타이밍에 푸시 알림 발송
3. **💰 비용 최적화:** 무분별한 쿠폰 살포를 막고, ROI 극대화를 위한 예산 정밀 배분

👉 **단순 예측이 아니라 '비즈니스 의사결정'을 돕는 도구입니다.**
            """,
            "tags": ["실무적용", "CRM", "비용절감"]
        }
    ],
    "데이터 및 평가지표": [
        {
            "question": "왜 정확도(Accuracy)를 성능 지표에서 제외했나요?",
            "answer": """
**A. 불균형 데이터(Imbalanced Data)의 함정**

이탈자(Churn)가 소수인 데이터에서 정확도는 '착시'를 일으킵니다. 
모델이 무조건 "이탈 안 함"이라고만 예측해도 정확도는 90% 이상 나올 수 있지만, **실제 이탈자는 한 명도 못 잡는 무용지물 모델**이 됩니다. 
우리는 이 함정을 피하고자 **Recall**과 **PR-AUC**에 집중했습니다.
            """,
            "tags": ["정확도", "데이터불균형", "함정"]
        },
        {
            "question": "왜 Macro F1-score를 핵심 지표로 사용했나요?",
            "answer": """
**A. 소수 클래스(이탈자)를 존중하기 위함**

**Macro F1-score**는 클래스별 데이터 양과 상관없이 **모든 클래스를 동등하게 평가**합니다.
반면 Micro F1이나 Accuracy는 데이터가 많은 쪽(비이탈)에 점수가 휘둘립니다.
이탈 예측의 핵심은 **소수인 '이탈자'를 얼마나 잘 맞추느냐**이므로, Macro F1이 가장 정직한 지표입니다.
            """,
            "tags": ["Macro F1", "평가지표", "핵심"]
        },
        {
            "question": "왜 ROC Curve보다 PR Curve와 Lift Chart를 강조했나요?",
            "answer": """
**A. 비즈니스 현실을 반영하는 정직한 도구**

* **ROC Curve:** 데이터 불균형이 심할 때 성능이 좋아 보이게 부풀려지는 경향이 있습니다.
* **PR Curve & Lift Chart:** 실제 타겟인 '이탈자'를 얼마나 정확하고 효율적으로 찾아내는지 가감 없이 보여줍니다. 

**예산 집행을 결정해야 하는 비즈니스 상황**에서는 PR Curve와 Lift Chart가 훨씬 신뢰할 수 있는 도구입니다.
            """,
            "tags": ["PR-AUC", "Lift Chart", "시각화"]
        }
    ],
    "모델 선정": [
        {
            "question": "왜 여러 모델 중 LightGBM을 최종 선택했나요?",
            "answer": """
우리는 **성능뿐만 아니라 해석 가능성, 안정성, 실서비스 적용 가능성**을 함께 고려했습니다.

**최종 선택 모델: LightGBM**
* ✅ **Macro F1-score 1위:** 불균형 데이터에서 가장 균형 잡힌 성능
* ✅ **속도:** XGBoost 대비 2배 이상 빠른 학습 및 추론 속도 (배포 유리)
* ✅ **해석 가능:** Feature Importance를 통해 마케팅 부서 설득 가능
            """,
            "tags": ["LightGBM", "최종선택", "SOTA"]
        },
        {
            "question": "다른 모델들과의 성능 차이는 얼마나 되나요?",
            "answer": """
**주요 모델 성능 비교 (Benchmark):**

| 모델 | Macro F1 | PR-AUC | 추론 속도 |
|------|----------|---------|-----------|
| **LightGBM** | **0.85** | **0.82** | ⚡ **빠름** |
| XGBoost | 0.83 | 0.81 | 보통 |
| Random Forest | 0.79 | 0.77 | 느림 |
| Logistic Reg | 0.72 | 0.70 | ⚡ 빠름 |

LightGBM이 **성능과 효율성(속도)을 모두 만족**시키는 최적의 선택이었습니다.
            """,
            "tags": ["벤치마크", "성능비교"]
        }
    ],
    "엔지니어링 및 실무": [
        {
            "question": "모델이 왜 이 고객을 이탈자로 예측했는지 설명할 수 있나요?",
            "answer": """
**A. XAI (Explainable AI) 적용**

네, 가능합니다. 저희는 단순 블랙박스 모델을 지양합니다.
**SHAP**이나 **Feature Importance** 분석을 통해 *"최근 접속일수 급감"*, *"결제 금액 감소"* 같은 구체적인 원인을 제시합니다.
이는 마케팅 부서가 **"왜 이 사람에게 쿠폰을 줘야 하는지"**에 대한 설득력 있는 근거가 됩니다.
            """,
            "tags": ["XAI", "설명가능성", "SHAP"]
        },
        {
            "question": "왜 변수를 더 많이 넣지 않고 14~22개만 사용했나요?",
            "answer": """
**A. 과적합(Overfitting) 방지 및 효율성**

변수가 무작정 많다고 성능이 오르지 않습니다. 오히려 노이즈가 되어 모델을 방해합니다.
도메인 지식과 상관관계 분석을 통해 **이탈과 가장 밀접한 '핵심 변수'만 엄선**하는 Feature Selection 과정을 거쳤습니다.
이를 통해 **가볍지만 강력하고, 유지보수가 쉬운 모델**을 만들었습니다.
            """,
            "tags": ["Feature Selection", "최적화", "과적합"]
        },
        {
            "question": "왜 CSV가 아니라 Parquet 파일인가요?",
            "answer": """
**A. 속도와 용량의 최적화**

* **용량:** CSV 대비 약 1/4 수준으로 압축됩니다.
* **속도:** 열 지향(Columnar) 저장 방식이라 대시보드 로딩 속도가 획기적으로 빠릅니다.
대용량 데이터 환경에서도 **스트림릿 앱이 지연 없이 실시간으로 작동**하게 하기 위한 기술적 선택입니다.
            """,
            "tags": ["Parquet", "데이터엔지니어링", "최적화"]
        },
        {
            "question": "학습 데이터에 없는 새로운 데이터가 들어오면 어떻게 되나요?",
            "answer": """
**A. 일반화(Generalization) 및 재학습 파이프라인**

1. **StandardScaler:** 데이터 스케일링을 통해 수치 크기에 편향되지 않도록 설계했습니다.
2. **Retraining:** 향후 데이터 분포 변화(Data Drift)에 대응하기 위해, 주기적으로 최신 데이터를 반영해 모델을 업데이트하는 **재학습 파이프라인**을 고려하고 있습니다.
            """,
            "tags": ["일반화", "재학습", "MLOps"]
        }
    ]
}

# ==============================================================================
# FAQ 렌더링 로직
# ==============================================================================
def render_faq(category_name, faqs):
    # 카테고리 헤더 표시
    st.markdown(f"""
    <div class="category-title" style="margin-top: 20px; margin-bottom: 10px; font-size: 1.2rem; font-weight: bold; color: #333;">
        {category_name} <span style="background-color: #f3f4f6; color: #666; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem;">{len(faqs)}개</span>
    </div>
    """, unsafe_allow_html=True)
    
    for idx, faq in enumerate(faqs):
        # 검색 필터링 로직
        if search_query:
            query = search_query.lower()
            if query not in faq["question"].lower() and query not in faq["answer"].lower():
                continue
        
        # 아코디언(Expander) 생성
        # 검색어가 있거나, 첫 번째 항목일 경우 자동으로 열리게 설정 (선택 사항)
        is_expanded = (idx == 0 and not search_query) or (search_query != "")
        
        with st.expander(f"Q. {faq['question']}", expanded=False):
            st.markdown(faq["answer"])
            
            # 태그 표시
            if "tags" in faq:
                st.write("") # 여백
                tags_html = " ".join([
                    f'<span style="background:#e0e7ff; color:#3730a3; padding:4px 10px; border-radius:12px; font-size:0.75rem; margin-right:6px; font-weight:600;">#{tag}</span>' 
                    for tag in faq["tags"]
                ])
                st.markdown(tags_html, unsafe_allow_html=True)

# 메인 로직 실행
if selected_category == "전체":
    for cat_name, items in faq_data.items():
        render_faq(cat_name, items)
else:
    if selected_category in faq_data:
        render_faq(selected_category, faq_data[selected_category])