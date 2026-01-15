import streamlit as st
import time
import plotly.graph_objects as go

# ===================== 페이지 설정 (반드시 최상단) =====================
st.set_page_config(
    page_title="SKN23_2nd_3Team",
    layout="wide"
)

# ===== util 파일 불러오기 =======
from utils.ui import apply_base_layout, hide_sidebar

apply_base_layout()
hide_sidebar()

# ------------------ CSS ------------------
# 색상 테마 정의
primary_red = "#dd2e1f"
# [수정] 타이틀용 진한 그라데이션 색상 (흰색 느낌 제거)
gradient_end = "#ffdff6" 
# 기존 연한 배경용 색상
light_red_bg_start = "#fef2f2"
light_red_bg_end = "#ffe4e6"
light_red_border = "#fecaca"

st.markdown(f"""
<style>
    /* 전체 배경 */
    .main {{
        background: linear-gradient(135deg, #fdfafa 0%, #e2e8f0 100%);
    }}
    
    .block-container {{
        padding-top: 2rem;
    }}

    /* [수정] 메인 타이틀: 흰색 그라데이션 제거 & 그림자 적용 */
    .main-title {{
        font-size: 3rem;
        font-weight: 900;
        /* 흰색(#ffdff6) 대신 진한 붉은색(#be123c)으로 변경하여 선명하게 */
        background: linear-gradient(135deg, {primary_red} 20%, {gradient_end} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 30px;
        text-align: left;
    }}

    /* Navigation Cards */
    .nav-card {{
        background: white;
        border-radius: 20px;
        padding: 35px 25px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        border: 2px solid #e2e8f0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        position: relative;
        overflow: hidden;
        margin-bottom: 15px;
    }}
    
    .nav-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, {primary_red}, {gradient_end});
        transform: scaleX(0);
        transition: transform 0.3s ease;
    }}
    
    .nav-card:hover {{
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(221, 46, 31, 0.12);
        border-color: {primary_red};
    }}
    
    .nav-card:hover::before {{
        transform: scaleX(1);
    }}
    
    .nav-icon {{
        font-size: 3rem;
        margin-bottom: 15px;
    }}
    
    .card-title {{
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 8px;
    }}
    
    .card-desc {{
        font-size: 0.95rem;
        color: #64748b;
        line-height: 1.5;
    }}

    /* Dashboard Section */
    .dashboard-header {{
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 15px;
    }}

    /* Progress Section Container */
    .progress-container {{
        background: linear-gradient(135deg, {light_red_bg_start} 0%, {light_red_bg_end} 100%);
        border-radius: 12px;
        padding: 12px 15px;
        margin-bottom: 15px;
        border: 2px solid {light_red_border};
    }}
    .stProgress > div:last-child > div > div > div {{
        background-color: {primary_red} !important;
    }}

    /* Streamlit 프로그레스 바 색상 강제 적용 */
    div[data-testid="stProgressBar"] > div > div > div > div {{
        background-color: {primary_red} !important;
    }}
    
    div[data-testid="stProgressBar"] > div > div {{
        background-color: #fce7f3 !important; 
    }}

    /* KPI Cards */
    .kpi-card {{
        background: white;
        border-radius: 12px;
        padding: 15px 12px;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    
    .kpi-card:hover {{
        border-color: {primary_red};
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(221, 46, 31, 0.15);
    }}
    
    .kpi-label {{
        font-size: 0.7rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }}
    
    .kpi-value {{
        font-size: 1.6rem;
        font-weight: 800;
        color: #1e293b;
    }}
    
    .kpi-delta {{
        font-size: 0.75rem;
        color: {primary_red};
        font-weight: 600;
        margin-top: 3px;
    }}

    /* Legend Box */
    .legend-box {{
        background: white;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 2px solid #e2e8f0;
        height: 100%;
    }}
    
    .legend-item {{
        padding: 12px;
        border-left: 4px solid;
        border-radius: 6px;
        margin-bottom: 10px;
        transition: all 0.3s ease;
    }}
    
    .legend-item:last-child {{
        margin-bottom: 0;
    }}
    
    .legend-item:hover {{
        transform: translateX(5px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }}
    
    .legend-item.m2 {{
        border-left-color: {primary_red};
        background: linear-gradient(to right, {light_red_bg_start} 0%, white 100%);
    }}
    
    .legend-item.other {{
        border-left-color: #94a3b8;
        background: linear-gradient(to right, #f8fafc 0%, white 100%);
    }}
    
    .legend-title {{
        font-weight: 700;
        font-size: 0.95rem;
        margin-bottom: 4px;
    }}
    
    .legend-title.m2 {{
        color: {primary_red};
    }}
    
    .legend-title.other {{
        color: #64748b;
    }}
    
    .legend-desc {{
        font-size: 0.75rem;
        color: #64748b;
        line-height: 1.3;
    }}

    /* Donut Chart Container */
    .donut-container {{
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-radius: 12px;
        padding: 15px;
        border: 2px solid #e2e8f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}

    /* Streamlit Button Override */
    .stButton > button {{
        width: 100%;
        border: none !important;
        background: none !important;
        padding: 0 !important;
        box-shadow: none !important;
    }}
    
    .stButton > button:hover {{
        border: none !important;
        background: none !important;
        box-shadow: none !important;
    }}
    
    .stButton > button:focus {{
        border: none !important;
        background: none !important;
        box-shadow: none !important;
    }}
</style>
""", unsafe_allow_html=True)

# ===================== 메인 레이아웃 =====================
st.markdown('<div class="main-title">E-commerce Churn Prediction</div>', unsafe_allow_html=True)
main1, space1, main2, space2 = st.columns([1, 0.1, 1, 0.1])

# ========== 왼쪽: 네비게이션 카드 ==========
with main1:
    # 첫 번째 행
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-icon">⚖️</div>
            <div class="card-title">Model</div>
            <div class="card-desc">모델 학습 & 평가</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Model", use_container_width=True, key="btn1"):
            st.switch_page("pages/2_Model_Compare.py")


    with col2:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-icon">📊</div>
            <div class="card-title">Report</div>
            <div class="card-desc">결과 분석 리포트</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Report", use_container_width=True, key="btn2"):
            st.switch_page("pages/3_Report_Download.py")

    # 두 번째 행
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-icon">💡</div>
            <div class="card-title">Predict</div>
            <div class="card-desc">예측 결과</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Predict", use_container_width=True, key="btn3"):
                st.switch_page("pages/4_Predict.py")




    with col4:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-icon">❓</div>
            <div class="card-title">Q&A</div>
            <div class="card-desc">질의응답</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Q&A", use_container_width=True, key="btn4"):
            st.switch_page("pages/5_FAQ_QnA.py")

# ========== 오른쪽: 대시보드 ==========
with main2:
    st.markdown('<div class="dashboard-header">Processing E-commerce Logs</div>', unsafe_allow_html=True)

    # Progress Section
    progress_container = st.container()
    with progress_container:
        # progress_container div가 스타일을 먹지만, st.progress 자체 스타일은 CSS로 오버라이드 됨
        progress_bar = st.progress(0)
        progress_text = st.empty()

    # KPI 영역
    c1, c2, c3, c4 = st.columns(4)

    kpi_total = c1.empty()
    kpi_users = c2.empty()
    kpi_m2 = c3.empty()
    kpi_prauc = c4.empty()

    st.markdown("<br>", unsafe_allow_html=True)

    # 도넛 + 범례 영역
    space, Donut1, Donut2 = st.columns([0.5,2, 1])

    with Donut1:
        donut_placeholder = st.empty()

    with Donut2:
        st.markdown(f"""
            <div class="legend-item m2">
                <div class="legend-title m2">🔴 m2</div>
                <div class="legend-desc">이탈 위험 높은 관심 유저</div>
            </div>
            <div class="legend-item other">
                <div class="legend-title other">⚪ 기타(m0/m1)</div>
                <div class="legend-desc">정상 활동 일반 유저</div>
            </div>
        """, unsafe_allow_html=True)

    # ===================== 초기 KPI =====================
    kpi_total.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">Total Logs</div>
        <div class="kpi-value">3.78M</div>
        <div class="kpi-delta"><br></div>
    </div>
    """, unsafe_allow_html=True)

    kpi_users.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">Unique Users</div>
        <div class="kpi-value">600K</div>
        <div class="kpi-delta"><br></div>
    </div>
    """, unsafe_allow_html=True)

    # ===================== 애니메이션 로직 =====================
    TOTAL = 3_780_000
    MAX_M2 = 82
    START_PRAUC = 0.65
    END_PRAUC = 0.934

    for i in range(101):
        # ---- Progress ----
        progress_bar.progress(i)
        processed = int(TOTAL * i / 100)
        # 진행률 텍스트 색상도 붉은색으로 변경
        progress_text.markdown(
            f"""
            <div style='text-align: center; font-size: 0.9rem; color: {primary_red}; font-weight: 600; margin-bottom: 1rem;'>
                Processed <strong>{processed:,}</strong> / <strong>{TOTAL:,}</strong> logs
            </div>
            """,
            unsafe_allow_html=True
        )

        # ---- m2 KPI ----
        m2_pct = min(MAX_M2, i * (MAX_M2 / 100))
        kpi_m2.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">m2 Rate</div>
            <div class="kpi-value">{m2_pct:.1f}%</div>
            <div class="kpi-delta">↑ High Risk</div>
        </div>
        """, unsafe_allow_html=True)

        # ---- PR-AUC KPI ----
        prauc = START_PRAUC + (i / 100) * (END_PRAUC - START_PRAUC)
        kpi_prauc.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Best PR-AUC</div>
            <div class="kpi-value">{prauc:.3f}</div>
            <div class="kpi-delta">↑ +{(prauc - START_PRAUC):.3f}</div>
        </div>
        """, unsafe_allow_html=True)

        # ---- Donut Chart ----
        fig = go.Figure(go.Pie(
            # 1. [채워진 값, 나머지] 순서로 변경
            values=[m2_pct, 100 - m2_pct],
            hole=0.55,
            # 2. 색상 순서도 [새로운 붉은색, 회색]으로 변경
            marker=dict(colors=[primary_red, "#E0E0E0"]),
            # 3. 중요: 크기순 정렬을 막아야 애니메이션이 튀지 않음
            sort=False,
            # 4. 시계 방향으로 채워지도록 설정
            direction='clockwise',
            textinfo="none"
        ))

        fig.update_layout(
            width=260,
            height=260,
            showlegend=False,
            annotations=[dict(
                text=f"m2<br><b>{m2_pct:.1f}%</b>",
                x=0.5, y=0.5,
                showarrow=False,
                font_size=20,
                font=dict(color=primary_red)
            )],
            margin=dict(t=0, b=0, l=0, r=0)
        )

        donut_placeholder.plotly_chart(fig, use_container_width=False)

        time.sleep(0.05)