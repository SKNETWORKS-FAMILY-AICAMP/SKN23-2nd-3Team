import streamlit as st
import time
import plotly.graph_objects as go

# ===================== 페이지 설정 (반드시 최상단) =====================
st.set_page_config(layout="wide")

# 페이지 이름 설정
st.set_page_config(
    page_title="SKN23 2ND 3TEAM",
    layout="wide"
)

# ======= 사이드바 숨김 =========
st.markdown("""
<style>
[data-testid="stSidebar"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# ------------------ CSS ------------------
st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 40px;
}

.card {
    background: linear-gradient(135deg, #1f1f1f, #2c2c2c);
    padding: 40px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    cursor: pointer;
}

.card:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.6);
}

.card-title {
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 10px;
}

.card-desc {
    font-size: 16px;
    color: #bdbdbd;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    /* 메인 타이틀 스타일 */
    .main-title {
        font-size: 42px;
        font-weight: 900;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF9068);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .sub-title {
        font-size: 18px;
        color: #666;
        margin-bottom: 40px;
    }

    /* 네비게이션 카드 스타일 */
    .nav-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .nav-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        border-color: #FF4B4B;
    }
    
    /* KPI 박스 스타일 */
    .kpi-box {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ Title ------------------
import streamlit as st
import time
import plotly.graph_objects as go

main1, spacer, main2 = st.columns([1, 0.1, 1])

with main1:
    st.markdown('<div class="main-title">E-commerce 이탈자 예측</div>', unsafe_allow_html=True)

    # ------------------ Buttons ------------------
    col1, col2 = st.columns(2)

    col3, col4 = st.columns(2)

    with col1:
        if st.button("📈 Overview", use_container_width=True):
            st.switch_page("pages/1_Overview.py")     # 페이지 연결 : 1page.py로 이동
        st.markdown("""
        <div class="card">
            <div class="card-title">Overview</div>
            <div class="card-desc">프로젝트 개요</div>
        </div>
        """, unsafe_allow_html=True)



    with col2:
        if st.button("🤖 Model", use_container_width=True):
            st.switch_page("pages/2_Model_Compare.py")     # 페이지 연결 : 2page.py로 이동
        st.markdown("""
        <div class="card">
            <div class="card-title">ML/DL</div>
            <div class="card-desc">모델 학습 & 평가</div>
        </div>
        """, unsafe_allow_html=True)


    with col3:
        # 행 사이 간격 줌.
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📊 Analysis", use_container_width=True):
            st.switch_page("pages/3_Report_Download.py")     # 페이지 연결 : 3page.py로 이동
        st.markdown("""
        <div class="card">
            <div class="card-title">Analysis</div>
            <div class="card-desc">결과 분석 리포트</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        # 행 사이 간격 줌.
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✔️ Q&A", use_container_width=True):
            st.switch_page("pages/4_FAQ_QnA.py")
        st.markdown("""
        <div class="card">
            <div class="card-title">Q&A</div>
            <div class="card-desc">질의응답</div>
        </div>
        """, unsafe_allow_html=True)


with main2:
    # ===================== 메인 레이아웃 =====================
    st.subheader("Processing E-commerce Logs")

    progress_bar = st.progress(0)
    progress_text = st.empty()

    # KPI 영역
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)

    kpi_total = c1.empty()
    kpi_users = c2.empty()
    kpi_m2 = c3.empty()
    kpi_prauc = c4.empty()

    # 도넛 영역
    Donut1, Donut2 = st.columns([2, 1])

    with Donut1:
        donut_placeholder = st.empty()

    with Donut2:
        st.markdown("""
            <div style='padding: 10px; border-left: 4px solid #FF4B4B;
                        background-color: #FFF5F5; margin-bottom: 10px;'>
                <strong style='color: #FF4B4B;'>🔴 m2</strong><br/>
                <span style='font-size: 0.9em;'>관심 유저</span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div style='padding: 10px; border-left: 4px solid #E0E0E0;
                        background-color: #F9F9F9;'>
                <strong style='color: #666;'>⚪ 기타</strong><br/>
                <span style='font-size: 0.9em;'>일반 유저</span>
            </div>
        """, unsafe_allow_html=True)

    # ===================== 초기 KPI =====================
    kpi_total.metric("Total Logs", "3.78M")
    kpi_users.metric("Unique Users", "420K")

    # ===================== 애니메이션 로직 =====================
    TOTAL = 3_780_000
    MAX_M2 = 18.5
    START_PRAUC = 0.65
    END_PRAUC = 0.82

    for i in range(101):
        # ---- Progress ----
        progress_bar.progress(i)
        processed = int(TOTAL * i / 100)
        progress_text.markdown(
            f"📦 Processed **{processed:,} / {TOTAL:,} logs**"
        )

        # ---- m2 KPI ----
        m2_pct = min(MAX_M2, i * (MAX_M2 / 100))
        kpi_m2.metric("m2 Rate", f"{m2_pct:.1f}%", delta="+5%")

        # ---- PR-AUC KPI (점진적 증가) ----
        prauc = START_PRAUC + (i / 100) * (END_PRAUC - START_PRAUC)
        kpi_prauc.metric("Best PR-AUC", f"{prauc:.3f}")

        # ---- Donut Chart ----
        fig = go.Figure(go.Pie(
            values=[100 - m2_pct, m2_pct],
            hole=0.55,
            marker=dict(colors=["#E0E0E0", "#FF4B4B"]),
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
                font_size=20
            )],
            margin=dict(t=0, b=0, l=0, r=0)
        )

        donut_placeholder.plotly_chart(fig, use_container_width=False)

        time.sleep(0.05)