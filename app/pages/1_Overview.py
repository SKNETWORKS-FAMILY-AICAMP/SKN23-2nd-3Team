import streamlit as st

# ===== util 파일 불러오기 =======
from utils.ui import apply_base_layout, hide_sidebar, top_nav, overview_ui

st.set_page_config(layout="wide")

apply_base_layout()
hide_sidebar()
# =============================
# 상단 네비게이션바
top_nav()

# ==== 간격 조정 =====
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
        gap: 0.3rem !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# CSS 스타일링
overview_ui()


# ==============================
import streamlit as st

# 페이지 설정
st.set_page_config(page_title="Overview", layout="wide")


# Challenge Section
st.markdown("""
<div class="section-title">
    🚨 Challenge
</div>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="challenge-card">
        <div class="icon-box icon-box-red">📉</div>
        <div class="card-title">비효율적 비용 집행</div>
        <div class="card-text">
            이탈 위험이 낮은 충성 고객에게도 무분별하게 쿠폰을 지급하여 
            마케팅 예산이 낭비되고 있습니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="challenge-card">
        <div class="icon-box icon-box-orange">⚠️</div>
        <div class="card-title">데이터 누수 이슈</div>
        <div class="card-text">
            초기 모델링 과정에서 미래 정보(active_days > 30)가 포함된 
            치명적인 데이터 누수 문제를 발견했습니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="challenge-card">
        <div class="icon-box icon-box-yellow">🎯</div>
        <div class="card-title">클래스 불균형</div>
        <div class="card-text">
            전체 데이터 중 실제 휴면 고객(m2)은 18.5%에 불과해 
            정확한 예측이 어려운 상황입니다.
            <br>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)



# =======================
solution, _, impact = st.columns([1,0.05,1])
# Solution Section
with solution:
    st.markdown("""
    <div class="section-title">
        ✅ Solution
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="solution-box">
        <div class="solution-item">
            <div class="solution-number">1</div>
            <div>
                <div class="solution-title">Clean Data Pipeline</div>
                <div class="solution-text">
                    타임 윈도우(t-30 ~ t)를 엄격히 준수한 피처 엔지니어링으로 
                    데이터 누수를 원천 차단했습니다.
                </div>
            </div>
        </div>   
        <div class="solution-item">
            <div class="solution-number">2</div>
            <div>
                <div class="solution-title">Profit-Driven Modeling</div>
                <div class="solution-text">
                    단순 정확도가 아닌 PR-AUC와 Top-K Recall을 핵심 지표로 설정하여 
                    비즈니스 가치에 집중했습니다.
                </div>
            </div>
        </div>    
        <div class="solution-item">
            <div class="solution-number">3</div>
            <div>
                <div class="solution-title">Binary Classification Focus</div>
                <div class="solution-text">
                    복잡한 다중 분류 대신, 마케팅 액션이 필수적인 휴면 고객(m2) 탐지에 
                    집중하여 모델 성능을 극대화했습니다.
                </div>
            </div>
        </div>
        <div class="tech-stack">
            <div style="font-weight: 600; margin-bottom: 15px; color: #1e293b; display: flex; align-items: center; gap: 8px;">
                📊 핵심 기술 스택
            </div>
            <span class="tech-badge">XGBoost</span>
            <span class="tech-badge">Deep Learning</span>
            <span class="tech-badge">Feature Engineering</span>
            <span class="tech-badge">SMOTE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ======================
# --- Impact Section ---
with impact:
    st.markdown("""
    <div class="section-title">
        🏆 Impact
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">75%</div>
            <div class="metric-label">비용 절감</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">3.2x</div>
            <div class="metric-label">ROI 개선</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">92%</div>
            <div class="metric-label">Precision</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
        <div class="impact-note">
            💡 <strong>상세 분석</strong>은 Model Comparison 페이지에서 확인 가능합니다.
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
# --------------------------------------------------------------------------------
# 7. Key Metrics Summary (Flexbox/Grid로 통합하여 깔끔하게 수정)
# --------------------------------------------------------------------------------

    # CSS에 그리드 레이아웃 추가 (기존 CSS 아래에 추가되거나 통합됨)
    st.markdown("""
    <style>
        .metrics-container {
            display: grid;
            grid-template-columns: repeat(4, 1fr); /* 4등분 */
            gap: 20px;
            margin-top: 20px;
        }
        
        /* 모바일 대응: 화면 작아지면 2줄로 표시 */
        @media (max-width: 768px) {
            .metrics-container {
                grid-template-columns: repeat(2, 1fr);
            }
        }
    </style>
    """, unsafe_allow_html=True)

    # HTML 한 덩어리로 렌더링 (st.columns 제거)
    st.markdown("""
    <div class="metrics-box">
        <div class="metrics-title">📊 프로젝트 핵심 지표 요약</div>
        <div class="metrics-container">
            <div class="metric-item">
                <div class="metric-item-value" style="color: #fbbf24;">18.5%</div>
                <div class="metric-item-label">휴면 고객 비율</div>
            </div>
            <div class="metric-item">
                <div class="metric-item-value" style="color: #10b981;">5 EA</div>
                <div class="metric-item-label">비교 모델 수</div>
            </div>
            <div class="metric-item">
                <div class="metric-item-value" style="color: #3b82f6;">Top 5%</div>
                <div class="metric-item-label">최적 타겟팅 비율</div>
            </div>
            <div class="metric-item">
                <div class="metric-item-value" style="color: #a855f7;">PR-AUC</div>
                <div class="metric-item-label">핵심 평가 지표</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)