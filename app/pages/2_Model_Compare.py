import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os

# --------------------------------------------------------------------------------
# 1. 페이지 설정
# --------------------------------------------------------------------------------
st.set_page_config(page_title="Top-K 모델 성능 비교", page_icon="⚖️", layout="wide")

# ===== util 파일 불러오기 =======
from utils.ui import apply_base_layout, hide_sidebar, top_nav, apply_tooltip_style, model_tooltip, model_ui

apply_base_layout()
hide_sidebar()
top_nav()
apply_tooltip_style()
model_ui()

# ==== 간격 조정 ====
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
        gap: 0.7rem !important;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------------
# 3. 데이터 준비 (Mock Data)
# --------------------------------------------------------------------------------
@st.cache_data
def get_mock_data():
    np.random.seed(42)
    n = 2000
    y_true = np.random.choice([0, 1], size=n, p=[0.82, 0.18])
    
    def gen_score(base_acc, noise):
        return np.clip(y_true * base_acc + np.random.rand(n) * noise, 0, 1)

    df = pd.DataFrame({
        'actual': y_true,
        # --- ML Models ---
        'Logistic Regression': gen_score(0.40, 0.60),
        'Random Forest': gen_score(0.55, 0.45),
        'Decision Tree': gen_score(0.30, 0.70),
        'XGBoost': gen_score(0.75, 0.25),
        'LightGBM': gen_score(0.72, 0.28),
        'HistGradientBoosting': gen_score(0.70, 0.30),
        'ExtraTrees': gen_score(0.65, 0.35),
        # --- DL Models ---
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

# ⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️⭐️

# --------------------------------------------------------------------------------
# 4. Top-K 지표 계산 로직
# --------------------------------------------------------------------------------
# 5 15 30만 
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


# --------------------------------------------------------------------------------
# 5. 메인 화면 구성
# --------------------------------------------------------------------------------

st.title("⚖️ Model Performance Comparison")

# 레이아웃 정의
select, divider, _, compare = st.columns([1.5, 0.1, 0.1, 6])

# ==============================================================================
# [수정됨] 왼쪽 사이드바 영역 (정확히 이 컬럼만 회색으로!)
# ==============================================================================
with select:
    # 🎨 CSS 수정: 'stVerticalBlock'이 아니라 'column' 자체를 타겟팅
    st.markdown("""
    <style>
        /* 1. data-testid="column" : 스트림릿의 컬럼을 의미합니다. */
        /* 2. :has(div.gray-background) : 내부에 'gray-background'라는 표식이 있는 컬럼만 찾습니다. */
        div[data-testid="column"]:has(div.gray-background) {
            background-color: #f5f7f9; /* 아주 연한 회색 (취향껏 #f0f2f6 등으로 변경 가능) */
            border-radius: 15px;       /* 둥근 모서리 */
            padding: 20px;             /* 안쪽 여백 */
            box-shadow: 2px 2px 10px rgba(0,0,0,0.05); /* 살짝 그림자 줘서 붕 떠보이게 */
        }
    </style>
    <div class="gray-background"></div>
    """, unsafe_allow_html=True)
    
    st.markdown("##### 🛠️ Model Selection")
    
    # --- [왼쪽] Model A 설정 ---
    with st.container(border=True):
        st.markdown('<div class="section-header" style="color:#1f77b4;">🔵 Model A (Left)</div>', unsafe_allow_html=True)
        cat_a = st.radio("Category", ["ML", "DL"], key="cat_a", horizontal=True)
        model_a = st.selectbox("Select Model", MODEL_CATS[cat_a], key="model_a")

    # --- [오른쪽] Model B 설정 ---
    with st.container(border=True):
        st.markdown('<div class="section-header" style="color:#d62728;">🔴 Model B (Right)</div>', unsafe_allow_html=True)
        cat_b = st.radio("Category", ["ML", "DL"], key="cat_b", horizontal=True, index=1)
        default_idx_b = 1 if len(MODEL_CATS[cat_b]) > 1 else 0
        model_b = st.selectbox("Select Model", MODEL_CATS[cat_b], index=default_idx_b, key="model_b")

with divider:
    st.markdown("""
    <style>
    @media (max-width: 768px) {
        .vertical-divider {
            display: none;
        }
    }
    </style>

    <div class="vertical-divider"
         style="height: 700px; width: 0.1px; background-color: #d1d5db; margin: auto;">
    </div>
    """, unsafe_allow_html=True)


with compare :
    st.markdown("비교할 **두 모델**을 선택하고 **Top-K(상위 N%)** 범위를 설정하세요.")
    # ================================================================================
    # [섹션 2] 슬라이더 컨트롤 (납작한 디자인)
    # ================================================================================
    with st.container(border=True):
        st.markdown("### Target Audience & ROI Simulation")
        col_s1, col_s2 = st.columns([4, 1], gap="medium")

        with col_s1:
            k_percent = st.slider(
                "🎯 Top-K 분석 범위 설정 (%)", 
                min_value=1, max_value=30, value=5, step=1,
                help="이탈 확률 상위 N% 유저를 타겟팅합니다."
            )
            
            # 지표 계산 실행
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

            st.write("")

        with col_s2:
            n_targets = int(len(df) * (k_percent/100))
            st.metric("Total Targets", f"{n_targets:,}", delta="Top-K Count")

    # st.divider()
    st.write("")
    # ================================================================================
    # [섹션 3] 비교 결과 상세 (Radar Chart + Metrics)
    # ================================================================================
    col_left, col_mid_res, col_right = st.columns([1, 0.2, 1])

    # --- [왼쪽 결과] Model A ---
    with col_left:
        st.markdown(
            f"<div class='compare-header'>🔵 {model_tooltip(model_a, '#1f77b4')}</div>",
            unsafe_allow_html=True
        )
        st.info(f"Category: {cat_a}")

        c1, c2, c3 = st.columns(3)
        
        # Precision (Delta: Model A - Model B)
        c1.metric(
            label="Precision", 
            value=f"{prec_a:.1%}", 
            delta=f"{prec_a - prec_b:.1%}"
        )
        
        # Recall (Delta: Model A - Model B)
        c2.metric(
            label="Recall", 
            value=f"{rec_b:.1%}", 
            delta=f"{rec_a - rec_b:.1%}"
        )
        
        # Lift (Delta: Model A - Model B)
        c3.metric(
            label="Lift", 
            value=f"{lift_a:.2f}x", 
            delta=f"{lift_a - lift_b:.2f}x"
        )

        # Radar Chart A
        fig_a = go.Figure(data=go.Scatterpolar(
            r=[prec_a, rec_a, lift_a/5], # Lift는 스케일 조정 (시각화용)
            theta=['Precision', 'Recall', 'Lift'],
            fill='toself', 
            name=model_a, 
            line_color='#1f77b4'
        ))
        fig_a.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])), 
            showlegend=False, 
            height=250, 
            margin=dict(t=20, b=20, l=40, r=40)
        )
        st.plotly_chart(fig_a, use_container_width=True)

    # --- [가운데 결과] VS 배지 (Large) ---
    with col_mid_res:
        st.markdown("<div class='vs-badge-large'>VS</div>", unsafe_allow_html=True)

    # --- [오른쪽 결과] Model B ---
    with col_right:
        st.markdown(
            f"<div class='compare-header'>🔴 {model_tooltip(model_b, '#d62728')}</div>",
            unsafe_allow_html=True
        )
        st.error(f"Category: {cat_b}") # 빨간색 스타일 박스

    
        c1, c2, c3 = st.columns(3)
        
        # Precision (Delta: Model B - Model A)
        c1.metric(
            label="Precision", 
            value=f"{prec_b:.1%}", 
            delta=f"{prec_b - prec_a:.1%}"
        )
        
        # Recall (Delta: Model B - Model A)
        c2.metric(
            label="Recall", 
            value=f"{rec_b:.1%}", 
            delta=f"{rec_b - rec_a:.1%}"
        )
        
        # Lift (Delta: Model B - Model A)
        c3.metric(
            label="Lift", 
            value=f"{lift_b:.2f}x", 
            delta=f"{lift_b - lift_a:.2f}x"
        )

        # Radar Chart B
        fig_b = go.Figure(data=go.Scatterpolar(
            r=[prec_b, rec_b, lift_b/5], # Lift는 스케일 조정 (시각화용)
            theta=['Precision', 'Recall', 'Lift'],
            fill='toself', 
            name=model_b, 
            line_color='#d62728' # 빨간색 (Model B 테마)
        ))
        fig_b.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])), 
            showlegend=False, 
            height=250, 
            margin=dict(t=20, b=20, l=40, r=40)
        )
        st.plotly_chart(fig_b, use_container_width=True)