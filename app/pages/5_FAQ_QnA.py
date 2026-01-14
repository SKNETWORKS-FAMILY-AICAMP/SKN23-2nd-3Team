import streamlit as st
from utils.ui import apply_base_layout, hide_sidebar, top_nav, QnA_ui

# ===================== 페이지 설정 =====================
st.set_page_config(layout="wide", page_title="FAQ & QnA")

# UI 유틸리티 적용
apply_base_layout()
hide_sidebar()
top_nav()
QnA_ui()

# ============ CSS 스타일링 (간격 및 디자인) =============
st.markdown("""
<style>
    /* 기본 레이아웃 조정 */
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

    /* 카테고리 뱃지 스타일 */
    .category-badge {
        background-color: #f3f4f6; 
        color: #4b5563; 
        padding: 2px 8px; 
        border-radius: 12px; 
        font-size: 0.8rem; 
        font-weight: 500;
        margin-left: 8px;
    }

    /* 태그 스타일 */
    .tag-span {
        background: #e0e7ff; 
        color: #3730a3; 
        padding: 4px 10px; 
        border-radius: 12px; 
        font-size: 0.75rem; 
        margin-right: 6px; 
        font-weight: 600;
        display: inline-block;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# # ======= Hero Section (타이틀) =======
# st.markdown("""
# <div style="padding-bottom: 0px;">
#     <h1 style="
#         font-family: 'Helvetica Neue', sans-serif;
#         font-weight: 900;
#         font-size: 3rem;
#         background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         margin: 0;
#         padding-bottom: 5px;
#         padding-top: 10px;
#     ">
#     ⚡ Q&A
#     </h1>
#     <p style="
#         font-size: 1.1rem;
#         color: #6c757d;
#         margin: 0;
#         font-weight: 500;
#         padding-bottom: 15px;
#     ">
#         자주 묻는 질문(FAQ) 및 프로젝트 핵심 요약
#     </p>
# </div>
# """, unsafe_allow_html=True)

 # 제목
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;600;800&display=swap');
    
    .dashboard-header {
        position: relative;
        padding: 2.5rem 0 2rem 0;
        background: white;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 2rem;
    }
    
    .header-content {
        position: relative;
        z-index: 1;
    }
    
    .main-title {
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 800;
        font-size: 2.5rem;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -0.5px;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .subtitle {
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 1.1rem;
        color: #6b7280;
        margin: 0.75rem 0 0 0;
        font-weight: 500;
        letter-spacing: -0.2px;
        animation: fadeInUp 0.6s ease-out 0.1s both;
    }
    
    .accent-line {
        width: 60px;
        height: 4px;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        border-radius: 2px;
        margin-top: 1rem;
        animation: fadeInUp 0.6s ease-out 0.2s both;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
</style>

<div class="dashboard-header">
    <div class="header-content">
        <h1 class="main-title">FAQ & QnA</h1>
        <p class="subtitle">자주 묻는 질문(FAQ) 및 프로젝트 핵심 요약</p>
        <div class="accent-line"></div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ===== 검색 기능 ======
st.markdown('<div class="search-container">', unsafe_allow_html=True)
search_query = st.text_input(
    "🔍 FAQ 검색", 
    placeholder="키워드를 검색해보세요 (예: MLP, Parquet, Lift, 타겟팅)",
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

# 카테고리 필터
categories = ["전체", "프로젝트 기획 및 철학", "모델링 방향 및 문제 정의", "분석 결과 평가 및 지표 선택", "실제 서비스 적용"]
selected_category = st.radio(
    "카테고리 선택",
    categories,
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")

# ==============================================================================
# FAQ 데이터 (요청하신 내용 반영)
# ==============================================================================
faq_data = {
    "프로젝트 기획 및 철학": [
        {
            "question": "수많은 주제 중 왜 ‘고객 이탈 예측’을 선정했나요?",
            "answer": """
<strong>A. 비용 효율성 및 선제적 방어</strong><br><br>
이커머스 시장에서는 <strong>신규 고객을 유치하는 비용이 기존 고객을 유지하는 비용보다 훨씬 큽니다.</strong><br>
본 프로젝트는 전자상거래 로그 데이터를 분석하여, 휴면(m2) 상태로 전환될 가능성이 높은 고객을 사전에 식별하고 마케팅 예산을 효율적으로 집행하기 위해 기획되었습니다.
            """,
            "tags": ["기획의도", "ROI", "비용효율"]
        },
        {
            "question": "왜 고객을 단계로 나누지 않고 ‘상위 K% 이탈 가능 고객’ 방식으로 접근했나요?",
            "answer": """
<strong>A. 마케팅 실행력(Actionability) 최적화</strong><br><br>
고객은 브랜드 유지, 전환(Switch), 휴면 등 다양한 맥락을 가집니다. 단순한 단계 구분보다는,<br>
점수(Probability) 기반으로 정렬하여 <strong>상위 K% 이탈 가능 고객</strong>을 선별하는 방식이 
실제 마케팅 현장에서 예산 범위에 맞춰 타겟팅하기에 가장 적합하다고 판단했습니다.
            """,
            "tags": ["Top-K", "스코어링", "타겟팅"]
        },
        {
            "question": "왜 본 프로젝트에서는 휴면 고객을 이탈 고객으로 정의했나요?",
            "answer": """
<strong>A. 실질적인 관리 필요 시점</strong><br><br>
장기간 활동이 없는 고객은 별도의 액션이 없으면 재활성화될 확률이 급격히 낮아집니다.<br>
따라서 본 프로젝트에서는 <strong>'휴면 상태 진입'</strong>을 실질적인 이탈로 간주하고, 이 시점을 집중 관리해야 할 타이밍으로 정의했습니다.
            """,
            "tags": ["휴면고객", "이탈정의", "골든타임"]
        }
    ],
    "모델링 방향 및 문제 정의": [
        {
            "question": "왜 다중 분류가 아닌 휴면(m2) 여부 이진 분류로 문제를 정의했나요?",
            "answer": """
<strong>A. 명확한 액션 플랜 수립</strong><br><br>
실제 마케팅 액션은 '휴면 예정자'인가 '아닌가'에 따라 명확히 갈립니다.<br>
문제의 복잡도를 줄이고 타겟팅 정확도를 높이기 위해, <strong>휴면(m2) vs 비휴면(m0 + m1)</strong>의 <strong>이진 분류(Binary Classification)</strong> 문제로 단순화하여 접근했습니다.
            """,
            "tags": ["이진분류", "문제단순화", "타겟팅"]
        },
        {
            "question": "본 프로젝트에서는 어떤 머신러닝·딥러닝 모델들을 사용했나요?",
            "answer": """
<strong>A. ML 3종 및 DL(MLP) 고도화 모델 비교</strong><br><br>
<strong>1. 머신러닝 (Baseline):</strong> Logistic Regression(선형 기준), LightGBM, HGB (트리 기반 성능)<br>
<strong>2. 딥러닝 (MLP):</strong> 정형 데이터의 비선형 관계 학습을 위해 3단계로 고도화<br>
&nbsp;&nbsp; • <strong>Base:</strong> 기본 3층 신경망<br>
&nbsp;&nbsp; • <strong>Enhance:</strong> 배치 정규화 + 드롭아웃 적용<br>
&nbsp;&nbsp; • <strong>Advanced:</strong> ResNet 구조 + Focal Loss 적용
            """,
            "tags": ["LightGBM", "MLP", "ResNet", "FocalLoss"]
        },
        {
            "question": "최종 모델은 무엇이며, 선정 이유는 무엇인가요?",
            "answer": """
<strong>A. 최종 모델: MLP_advanced (Deep Learning)</strong><br><br>
<strong>[선정 과정]</strong><br>
머신러닝 모델은 기준선(Baseline)으로 활용했으며, 정형 피처 간의 복잡한 비선형 관계를 학습하기 위해 딥러닝(MLP)을 고도화했습니다.<br>
검증(Validation) 단계에서 <strong>Recall@10%를 최우선으로 최대화</strong>하는 전략을 사용했습니다.<br><br>
<strong>[선정 이유]</strong><br>
MLP_advanced 모델은 <strong>상위 5% 고객군에서 약 96.8%</strong>, <strong>상위 30% 고객군에서 약 94.7%</strong>의 높은 Precision을 보였습니다.<br>
이는 제한된 예산 내에서 이탈 위험군을 정밀하게 타격해야 하는 비즈니스 목표를 가장 잘 충족하는 결과입니다.
            """,
            "tags": ["최종모델", "성능평가", "Recall@10%", "Precision"]
        }
    ],
    "분석 결과 평가 및 지표 선택": [
        {
            "question": "왜 정확도(Accuracy)를 주요 성능 지표로 사용하지 않았나요?",
            "answer": """
<strong>A. 데이터 불균형으로 인한 착시 방지</strong><br><br>
휴면(m2) 유저 비율이 높은 데이터 특성상, 모델이 무조건 '휴면'이라고 예측해도 Accuracy는 높게 나옵니다.<br>
이러한 성능 과대평가를 막기 위해 Accuracy 대신 <strong>Precision, Recall, Lift</strong> 등을 핵심 지표로 사용했습니다.
            """,
            "tags": ["데이터불균형", "Accuracy한계", "함정"]
        },
        {
            "question": "왜 Top-K 기반 지표를 핵심으로, PR-AUC는 보조로 활용했나요?",
            "answer": """
<strong>A. 타겟팅 목적 적합성</strong><br><br>
본 프로젝트의 목적은 전체 분류가 아닌 <strong>'상위 위험군 선별'</strong>입니다.<br>
따라서 <strong>Top-K(상위 5~30%)에서의 Precision·Recall·Lift</strong>를 실질적인 성과 지표로 삼았고, PR-AUC는 모델의 전반적인 학습 안정성을 확인하는 보조 지표로만 활용했습니다.
            """,
            "tags": ["Top-K", "PR-AUC", "평가기준"]
        },
        {
            "question": "왜 CSV가 아니라 Parquet 파일을 사용했나요?",
            "answer": """
<strong>A. 대용량 데이터 처리 속도 향상</strong><br><br>
<strong>Parquet</strong>는 열 지향(Columnar) 저장 방식으로, 대용량 데이터 읽기/쓰기 속도가 CSV 대비 월등히 빠릅니다.<br>
피처 엔지니어링 및 모델 학습 과정에서 발생하는 <strong>데이터 I/O 병목을 최소화</strong>하기 위해 채택했습니다.
            """,
            "tags": ["Parquet", "데이터엔지니어링", "최적화"]
        },
        {
            "question": "왜 임계값(Threshold)보다 Top-K 기준을 사용했나요?",
            "answer": """
<strong>A. 예산 기반의 유연한 의사결정</strong><br><br>
특정 임계값(0.5)으로 자르는 것보다, 위험도 점수 순으로 정렬하여 <strong>예산이 허용하는 범위(상위 K%)</strong>까지 끊어서 관리하는 것이 실무적으로 훨씬 합리적이고 유연하기 때문입니다.
            """,
            "tags": ["실무적용", "유연성", "예산관리"]
        },
        {
            "question": "왜 F1-score를 주요 지표로 사용하지 않았나요?",
            "answer": """
<strong>A. 랭킹(Ranking) 중심의 목표</strong><br><br>
F1-score는 임계값에 따른 분류 성능을 요약하는 지표입니다.<br>
하지만 본 프로젝트는 <strong>'누가 더 위험한가'</strong>를 줄 세워 상위권을 뽑는 것이 목표이므로, 순위 기반의 Top-K 지표가 더 적절하다고 판단했습니다.
            """,
            "tags": ["F1-score", "Ranking", "목적적합성"]
        }
    ],
    "실제 서비스 적용": [
        {
            "question": "이 분석 결과는 실제 서비스에서 어떻게 활용할 수 있나요?",
            "answer": """
<strong>A. 초개인화 마케팅 및 리텐션 관리</strong><br><br>
이탈 확률이 높은 <strong>상위 K% 고객 리스트</strong>를 추출하여 CRM 시스템에 연동할 수 있습니다.<br>
이들에게 <strong>시크릿 쿠폰, 복귀 환영 포인트, 맞춤형 푸시 메시지</strong> 등을 선제적으로 발송하여 이탈을 막고 매출을 방어하는 데 활용됩니다.
            """,
            "tags": ["CRM", "마케팅액션", "매출방어"]
        }
    ]
}

# ==============================================================================
# FAQ 렌더링 함수
# ==============================================================================
def render_faq(category_name, faqs):
    # 카테고리 헤더
    st.markdown(f"""
    <div style="margin-top: 30px; margin-bottom: 15px; display: flex; align-items: center;">
        <span style="font-size: 1.4rem; font-weight: 800; color: #1f2937;">{category_name}</span>
        <span class="category-badge">{len(faqs)} Q&A</span>
    </div>
    """, unsafe_allow_html=True)
    
    for faq in faqs:
        # 검색 필터링
        if search_query:
            query = search_query.lower()
            if query not in faq["question"].lower() and query not in faq["answer"].lower():
                continue
        
        # 아코디언 생성
        with st.expander(f"Q. {faq['question']}", expanded=False):
            # 답변 출력 (HTML 허용)
            st.markdown(faq["answer"], unsafe_allow_html=True)
            
            # 태그 출력
            if "tags" in faq:
                st.write("") # 간격
                tags_html = "".join([f'<span class="tag-span">#{tag}</span>' for tag in faq["tags"]])
                st.markdown(tags_html, unsafe_allow_html=True)

# ==============================================================================
# 메인 실행 로직
# ==============================================================================
if selected_category == "전체":
    for cat_name, items in faq_data.items():
        # 검색어가 있을 때는 해당 검색어가 포함된 항목이 있는 카테고리만 표시하거나
        # 전체를 돌면서 필터링된 결과가 있을 때만 헤더를 표시하는 로직이 필요할 수 있음
        # 여기서는 단순하게 전체 루프
        render_faq(cat_name, items)
else:
    if selected_category in faq_data:
        render_faq(selected_category, faq_data[selected_category])

# 하단 여백
st.markdown("<br><br>", unsafe_allow_html=True)