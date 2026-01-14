import streamlit as st
# ===== util 파일 불러오기 =======
from utils.ui import apply_base_layout, hide_sidebar, top_nav, QnA_ui

st.set_page_config(layout="wide", page_title="FAQ_QnA")

# UI 유틸리티 적용
apply_base_layout()
hide_sidebar()
top_nav()
QnA_ui()

# ============ 간격 조정 =============
st.markdown("""
<style>
    .block-container { 
        padding-top: 0.6rem !important;
        padding-bottom: 3rem; 
    }
    h1 {
        padding-top: 0rem !important;
        margin-top: -2rem !important;
    }
    div[data-testid="stVerticalBlock"] {
        gap: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================================

# ======= Hero Section =======
st.markdown("""
<div style="padding-bottom: 0px;">
    <h1 style="
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 900;
        font-size: 3rem;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        padding-bottom: 5px;
        padding-top: 10px;
    ">
    ⚡ Q&A
    </h1>
    <p style="
        font-size: 1.1rem;
        color: #6c757d;
        margin: 0;
        font-weight: 500;
        padding-bottom: 15px;
    ">
        FAQ & QnA
    </p>
</div>
""", unsafe_allow_html=True)
# st.divider()

# ===== 검색 기능 ======
st.markdown('<div class="search-container">', unsafe_allow_html=True)
search_query = st.text_input(
    "🔍 FAQ 검색", 
    placeholder="궁금한 키워드를 검색해보세요 (예: F1-score, LightGBM, Parquet)",
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

# 카테고리 필터 (데이터 구조에 맞춰 업데이트)
categories = ["전체", "프로젝트 기획 및 철학", "모델링 방향 및 문제 정의", "분석 결과 평가 및 지표 선택", "실제 서비스 적용 및 의사결정"]
selected_category = st.radio(
    "카테고리",
    categories,
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")

# ==============================================================================
# FAQ 데이터 구조화 (수정 완료)
# ==============================================================================
faq_data = {
    "프로젝트 기획 및 철학": [
        {
            "question": "수많은 주제 중 왜 '고객 이탈 예측'을 선정했나요?",
            "answer": """
<strong>A. 비용 효율성 및 선제적 관리</strong><br>
<br>
이커머스에서는 <strong>신규 고객 유치 비용</strong>이 기존 고객 유지 비용보다 큽니다.<br>
본 프로젝트는 전자상거래 로그 데이터를 기반으로 <strong>휴면(m2) 가능성이 높은 고객</strong>을 사전에 식별하여 수익성을 방어하고자 진행되었습니다.
            """,
            "tags": ["기획의도", "ROI", "리텐션"]
        },
        {
            "question": "왜 고객을 단계로 나누지 않고 '상위 K% 이탈 가능 고객' 방식으로 접근했나요?",
            "answer": """
<strong>A. 마케팅 실행력(Actionability) 강화</strong><br>
<br>
이커머스 고객은 브랜드 유지, 전환(Switch), 휴면 등 행동 맥락이 다양합니다.<br>
따라서 모호한 단계 구분보다는, 점수 기반으로 <strong>상위 K% 이탈 가능 고객</strong>을 선별하는 방식이 실제 <strong>타겟 마케팅</strong> 활용 관점에서 더 적합하다고 판단했습니다.
            """,
            "tags": ["Top-K", "마케팅", "타겟팅"]
        },
        {
            "question": "왜 본 프로젝트에서는 휴면 고객을 이탈 고객으로 정의했나요?",
            "answer": """
<strong>A. 실질적인 이탈 관리 시점</strong><br>
<br>
장기간 미활동 고객은 추가 액션이 없을 경우 <strong>재활성화 가능성</strong>이 낮아지는 경향이 있습니다.<br>
따라서 본 프로젝트에서는 <strong>휴면 상태 진입</strong>을 실질적인 이탈 관리 시점으로 정의했습니다.
            """,
            "tags": ["휴면고객", "이탈정의", "재활성화"]
        }
    ],
    "모델링 방향 및 문제 정의": [
        {
            "question": "휴면(m2) 고객 식별을 위해 다중 분류가 아닌 이진 분류를 선택한 이유는 무엇인가요?",
            "answer": """
<strong>A. 명확한 액션 플랜 수립</strong><br>
<br>
실제 마케팅 액션은 휴면 여부에 따라 명확히 구분되는 경우가 많습니다.<br>
이에 따라 본 프로젝트에서는 <strong>휴면(m2) vs 비휴면(m0, m1)</strong>의 <strong>이진 분류(Binary Classification)</strong>로 문제를 단순화하여 해결했습니다.
            """,
            "tags": ["이진분류", "문제정의", "단순화"]
        },
        {
            "question": "본 프로젝트에서는 어떤 머신러닝·딥러닝 모델들을 사용했나요?",
            "answer": """
<strong>A. ML 3종 및 DL(MLP) 모델 비교</strong><br>
<br>
• <strong>머신러닝(ML):</strong> Logistic Regression, LightGBM, HGB (3종)<br>
• <strong>딥러닝(DL):</strong> MLP(다층 퍼셉트론) 기반 딥러닝 모델 3종<br>
<br>
딥러닝 모델은 규제 기법과 구조적 변형을 단계적으로 적용하며 성능을 비교·실험했습니다.<br>
1. <strong>MLP_base:</strong> 기본 3층 신경망 (Baseline)<br>
2. <strong>MLP_enhance:</strong> Base + 배치 정규화 + 드롭아웃<br>
3. <strong>MLP_advanced:</strong> ResNet 구조 + Focal Loss
            """,
            "tags": ["모델선정", "LightGBM", "MLP", "ResNet"]
        }
    ],
    "분석 결과 평가 및 지표 선택": [
        {
            "question": "왜 정확도(Accuracy)를 주요 성능 지표로 사용하지 않았나요?",
            "answer": """
<strong>A. 성능 과대평가 방지</strong><br>
<br>
휴면(m2) 유저 비율이 높은 데이터 특성상, <strong>Accuracy(정확도)</strong>는 모델 성능을 과대평가할 위험이 있습니다.<br>
따라서 전체적인 예측 신뢰도는 <strong>PR-AUC</strong>로 확인하고, 실질적인 선별 성능은 <strong>Top-K 지표</strong>로 평가했습니다.
            """,
            "tags": ["정확도", "데이터불균형", "함정"]
        },
        {
            "question": "왜 Top-K 기반 Precision·Recall·Lift를 핵심으로, PR-AUC는 보조로 썼나요?",
            "answer": """
<strong>A. 타겟팅 목적에 부합하는 지표 선정</strong><br>
<br>
본 프로젝트의 목적은 <strong>상위 K% 고객 선별</strong>입니다.<br>
따라서 <strong>Top-K 기반 Precision·Recall·Lift</strong>를 핵심 지표로 삼아 타겟팅 성능을 검증하고, <strong>PR-AUC</strong>는 모델의 전반적인 분별력과 안정성을 확인하는 <strong>보조 지표</strong>로 활용했습니다.
            """,
            "tags": ["Top-K", "Lift", "PR-AUC"]
        },
        {
            "question": "왜 F1-score 및 macro·micro F1을 주요 지표로 쓰지 않았나요?",
            "answer": """
<strong>A. 임계값(Threshold) 의존성 탈피</strong><br>
<br>
F1 관련 지표들은 특정 <strong>임계값(0.5 등)</strong>을 기준으로 한 분류 성능을 요약합니다.<br>
점수 기반으로 고객을 정렬하여 <strong>상위 K%를 선별</strong>하는 본 프로젝트의 활용 목적을 직접적으로 반영하기 어렵다고 판단하여 제외했습니다.
            """,
            "tags": ["F1-score", "임계값", "평가기준"]
        },
        {
            "question": "왜 임계값(threshold)보다 Top-K 기준을 사용했나요?",
            "answer": """
<strong>A. 예산 및 리소스 기반 의사결정</strong><br>
<br>
임계값 기반 분류는 참고용으로만 활용했습니다.<br>
실무에서는 가용 예산에 맞춰 대상을 선정해야 하므로, 점수 기반 정렬 후 <strong>상위 K% 고객</strong>의 성과(Precision·Recall·Lift)를 확인하는 것이 훨씬 합리적입니다.
            """,
            "tags": ["Top-K", "랭킹", "실무적용"]
        },
        {
            "question": "왜 CSV가 아니라 Parquet 파일을 사용했나요?",
            "answer": """
<strong>A. 대용량 데이터 처리 효율성</strong><br>
<br>
<strong>Parquet</strong>는 열 지향(Columnar) 저장 방식으로 대용량 데이터 처리에 매우 효율적입니다.<br>
이를 통해 피처 생성 및 분석 과정에서 <strong>데이터 로딩 병목(I/O)</strong>을 획기적으로 줄일 수 있었습니다.
            """,
            "tags": ["Parquet", "최적화", "데이터엔지니어링"]
        }
    ],
    "실제 서비스 적용 및 의사결정": [
        {
            "question": "이 분석 결과는 실제 서비스에서 어떻게 활용할 수 있나요?",
            "answer": """
<strong>A. 정밀 타겟팅 및 마케팅 최적화</strong><br>
<br>
고객을 이탈 가능성 점수로 정렬하여 <strong>상위 K% 고객</strong>을 핵심 타겟으로 설정할 수 있습니다.<br>
이들을 대상으로 <strong>쿠폰 발송, 리마인드 메시지, 개인화 추천</strong> 등의 마케팅 액션을 우선 적용함으로써 마케팅 ROI를 극대화할 수 있습니다.
            """,
            "tags": ["CRM", "마케팅액션", "개인화"]
        }
    ]
}

# ==============================================================================
# FAQ 렌더링 로직 (unsafe_allow_html=True 적용)
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
        with st.expander(f"Q. {faq['question']}", expanded=False):
            # [수정됨] HTML 태그 렌더링 활성화
            st.markdown(faq["answer"], unsafe_allow_html=True)
            
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