import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# ==============================================================================
# 1. 페이지 설정 및 유틸 불러오기 (중복 제거됨)
# ==============================================================================
st.set_page_config(page_title="Top-K 모델 성능 비교", page_icon="⚖️", layout="wide")

try:
    # 툴팁 및 UI 함수들을 한 번만 Import
    from utils.ui import apply_base_layout, hide_sidebar, top_nav, apply_tooltip_style, model_tooltip, model_ui
    
    apply_base_layout()    # 레이아웃 적용
    hide_sidebar()         # 사이드바 숨김
    top_nav()              # 상단 네비게이션 (여기서 딱 한 번만 실행!)
    apply_tooltip_style()  # 툴팁 CSS 적용
    model_ui()             # 모델 UI 스타일 적용

except ImportError:
    st.error("utils/ui.py 파일을 찾을 수 없습니다.")
    st.stop()

# ==============================================================================
# 2. 추가 스타일링 (간격 조정)
# ==============================================================================
st.markdown("""
<style>
    .block-container { padding-top: 1rem !important; padding-bottom: 3rem; }
    h1 { padding-top: 0rem !important; margin-top: -1rem !important; }
    div[data-testid="stVerticalBlock"] { gap: 0.7rem !important; }
    
    /* Cutoff 정보 스타일 */
    .cutoff-info {
        background-color: #f8f9fa;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        font-family: 'Courier New', Courier, monospace;
        margin-top: 10px;
    }
    
    /* VS 배지 스타일 */
    .vs-badge-large {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        font-size: 24px;
        font-weight: bold;
        color: #6c757d;
        margin-top: 50px;
    }
    
    .compare-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. 데이터 준비 (Mock Data)
# ==============================================================================
@st.cache_data
def get_mock_data():
    np.random.seed(42)
    n = 2000
    y_true = np.random.choice([0, 1], size=n, p=[0.82, 0.18])
    
    def gen_score(base_acc, noise):
        return np.clip(y_true * base_acc + np.random.rand(n) * noise, 0, 1)

    df = pd.DataFrame({
        'actual': y_true,
        'Logistic Regression': gen_score(0.40, 0.60),
        'Random Forest': gen_score(0.55, 0.45),
        'Decision Tree': gen_score(0.30, 0.70),
        'XGBoost': gen_score(0.75, 0.25),
        'LightGBM': gen_score(0.72, 0.28),
        'HistGradientBoosting': gen_score(0.70, 0.30),
        'ExtraTrees': gen_score(0.65, 0.35),
        'DNN (MLP)': gen_score(0.68, 0.32),
        'TabNet': gen_score(0.60, 0.40),
        'Wide & Deep': gen_score(0.62, 0.38)
    })
    return df

df = get_mock_data()
BASE_CHURN_RATE = df['actual'].mean()

MODEL_CATS = {
    "ML": ["XGBoost", "LightGBM", "Random Forest", "Logistic Regression", "Decision Tree", "HistGradientBoosting", "ExtraTrees"],
    "DL": ["DNN (MLP)", "TabNet", "Wide & Deep"]
}

# ==============================================================================
# 4. Top-K 지표 계산 로직
# ==============================================================================
def calculate_metrics_at_k(df, model_col, k_percent):
    df_sorted = df.sort_values(by=model_col, ascending=False)
    top_k_count = int(len(df) * (k_percent / 100))
    if top_k_count < 1: top_k_count = 1
    
    cutoff_score = df_sorted.iloc[top_k_count - 1][model_col]
    target_group = df_sorted.head(top_k_count)
    
    precision = target_group['actual'].mean()
    captured_churners = target_group['actual'].sum()
    total_churners = df['actual'].sum()
    recall = captured_churners / total_churners if total_churners > 0 else 0
    lift = precision / BASE_CHURN_RATE if BASE_CHURN_RATE > 0 else 0
    
    return precision, recall, lift, cutoff_score

# ==============================================================================
# 5. 메인 화면 구성
# ==============================================================================

st.markdown("# ⚖️ Top-K(상위 N%) 구간별 모델 비교")
st.markdown("---")

# 레이아웃 정의
select, divider, _, compare = st.columns([1.5, 0.1, 0.1, 6])

# --- [왼쪽] 모델 선택 사이드 ---
with select:
    st.markdown("""
    <style>
        div[data-testid="column"]:has(div.gray-background) {
            background-color: #f5f7f9;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        }
    </style>
    <div class="gray-background"></div>
    """, unsafe_allow_html=True)
    
    st.markdown("##### 🛠️ 모델 선택")
    
    with st.container(border=True):
        st.markdown('<div style="color:#1f77b4; font-weight:bold;">🔵 Model A (Left)</div>', unsafe_allow_html=True)
        cat_a = st.radio("Category", ["ML", "DL"], key="cat_a", horizontal=True)
        model_a = st.selectbox("Select Model", MODEL_CATS[cat_a], key="model_a")

    with st.container(border=True):
        st.markdown('<div style="color:#d62728; font-weight:bold;">🔴 Model B (Right)</div>', unsafe_allow_html=True)
        cat_b = st.radio("Category", ["ML", "DL"], key="cat_b", horizontal=True, index=1)
        model_b = st.selectbox("Select Model", MODEL_CATS[cat_b], index=1 if len(MODEL_CATS[cat_b]) > 1 else 0, key="model_b")

# --- [중앙] 구분선 ---
with divider:
    st.markdown('<div style="height: 700px; width: 0.1px; background-color: #d1d5db; margin: auto;"></div>', unsafe_allow_html=True)

# --- [오른쪽] 비교 및 결과 ---
with compare:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("비교할 **두 모델**을 선택하고 **전략적 Top-K(상위 N%)** 구간을 설정하세요.")
    
    # [섹션 2] 4구간 전용 슬라이더
    with st.container(border=True):
        st.markdown("### Target Audience & ROI Simulation")
        col_s1, col_s2 = st.columns([4, 1], gap="medium")

        with col_s1:
            k_percent = st.select_slider(
                "🎯 Top-K 분석 범위 설정 (%)", 
                options=[5, 10, 15, 30],
                value=5,
                help="전략적 타겟팅 구간(5%, 10%, 15%, 30%) 중 하나를 선택하세요."
            )
            
            # 지표 계산
            prec_a, rec_a, lift_a, cut_a = calculate_metrics_at_k(df, model_a, k_percent)
            prec_b, rec_b, lift_b, cut_b = calculate_metrics_at_k(df, model_b, k_percent)
            
            # Cutoff 정보 표시
            st.markdown(f"""
            <div class='cutoff-info'>
                ✂️ <b>Cutoff Score:</b> 
                <span style='color:#1f77b4'>🔵 {model_a} > <b>{cut_a:.4f}</b></span> &nbsp;|&nbsp; 
                <span style='color:#d62728'>🔴 {model_b} > <b>{cut_b:.4f}</b></span> 
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")

        with col_s2:
            n_targets = int(len(df) * (k_percent/100))
            st.metric("Total Targets", f"{n_targets:,}", delta=f"Top {k_percent}%", help="타겟 유저 수")

    st.write("")
    
    # [섹션 3] 상세 결과 비교 (툴팁 적용됨!!)
    col_left, col_mid_res, col_right = st.columns([1, 0.2, 1])

    # --- Model A 결과 ---
    with col_left:
        # 🔥 여기가 핵심입니다! model_tooltip() 함수를 써야 툴팁이 뜹니다.
        # 이전 코드: f"🔵 {model_a}"
        # 수정 코드: f"🔵 {model_tooltip(model_a, '#1f77b4')}"
        st.markdown(
            f"<div class='compare-header'>🔵 {model_tooltip(model_a, '#1f77b4')}</div>", 
            unsafe_allow_html=True
        )
        st.info(f"Category: {cat_a}")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{prec_a:.1%}", delta=f"{prec_a - prec_b:.1%}")
        c2.metric("Recall", f"{rec_a:.1%}", delta=f"{rec_a - rec_b:.1%}")
        c3.metric("Lift", f"{lift_a:.2f}x", delta=f"{lift_a - lift_b:.2f}x")

        # Radar Chart A
        fig_a = go.Figure(data=go.Scatterpolar(
            r=[prec_a, rec_a, lift_a/5], 
            theta=['Precision', 'Recall', 'Lift'],
            fill='toself', name=model_a, line_color='#1f77b4'
        ))
        fig_a.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, height=280, margin=dict(t=30, b=30))
        st.plotly_chart(fig_a, use_container_width=True)

    # --- VS 배지 ---
    with col_mid_res:
        st.markdown("<div class='vs-badge-large'>VS</div>", unsafe_allow_html=True)

    # --- Model B 결과 ---
    with col_right:
        # 🔥 여기도 model_tooltip 적용 완료
        st.markdown(
            f"<div class='compare-header'>🔴 {model_tooltip(model_b, '#d62728')}</div>", 
            unsafe_allow_html=True
        )
        st.error(f"Category: {cat_b}")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{prec_b:.1%}", delta=f"{prec_b - prec_a:.1%}")
        c2.metric("Recall", f"{rec_b:.1%}", delta=f"{rec_b - rec_a:.1%}")
        c3.metric("Lift", f"{lift_b:.2f}x", delta=f"{lift_b - lift_a:.2f}x")

        # Radar Chart B
        fig_b = go.Figure(data=go.Scatterpolar(
            r=[prec_b, rec_b, lift_b/5], 
            theta=['Precision', 'Recall', 'Lift'],
            fill='toself', name=model_b, line_color='#d62728'
        ))
        fig_b.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, height=280, margin=dict(t=30, b=30))
        st.plotly_chart(fig_b, use_container_width=True)