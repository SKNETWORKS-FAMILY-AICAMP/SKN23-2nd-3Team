import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import json
from pathlib import Path

# ==============================================================================
# 1. 페이지 설정 (반드시 코드 최상단)
# ==============================================================================
st.set_page_config(page_title="Top-K 모델 성능 비교", page_icon="⚖️", layout="wide")

# ==============================================================================
# 2. 경로 자동 설정 로직
# ==============================================================================
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent

path_candidates = [
    project_root / "eval",             
    project_root / "models" / "eval",  
    current_file_path.parent / "eval", 
    Path("eval").resolve()             
]

EVAL_ROOT = None
for path in path_candidates:
    if path.exists():
        EVAL_ROOT = path
        break

if EVAL_ROOT is None:
    st.error(" 'eval' 폴더를 찾을 수 없습니다.")
    st.stop()

# ==============================================================================
# 3. 유틸 및 UI 불러오기
# ==============================================================================
try:
    from utils.ui import apply_base_layout, hide_sidebar, top_nav, apply_tooltip_style, model_tooltip, model_ui
    apply_base_layout()    
    hide_sidebar()         
    top_nav()              
    apply_tooltip_style()  
    model_ui()   

except ImportError:
    st.warning("⚠️ utils.ui 모듈을 찾을 수 없습니다. 기본 스타일로 실행합니다.")
    # 툴팁 함수가 없을 경우를 대비한 더미 함수
    def model_tooltip(name, color):
        return f"<span style='color:{color}'>{name}</span>"

# ==============================================================================
# 4. 스타일링 CSS (그림자 카드 + 정렬 수정 포함)
# ==============================================================================
st.markdown("""
<style>
    .block-container { padding-top: 1rem !important; padding-bottom: 3rem; }
    
    /* 헤더 스타일 */
    .compare-header {
        min-height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 10px;
        font-size: 1.5rem;
        font-weight: 800;
    }

    /* 지표 가운데 정렬 */
    [data-testid="stMetric"] { text-align: center !important; margin: auto; }
    [data-testid="stMetricLabel"] { justify-content: center !important; width: 100%; }
    [data-testid="stMetricValue"] { justify-content: center !important; width: 100%; }
    [data-testid="stMetricDelta"] { justify-content: center !important; width: 100%; }

    /* VS 배지 */
    .vs-badge-large {
        display: flex; align-items: center; justify-content: center;
        height: 100%; font-size: 24px; font-weight: bold; color: #6c757d;
        padding-top: 80px;
    }
    
    .cutoff-info {
        background-color: #f8f9fa;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        font-family: 'Courier New', Courier, monospace;
        margin-top: 10px;
        font-size: 0.9rem;
    }

    /* -----------------------------------------------------------------
       🔥 컨테이너 스타일 (Shadow Card)
       ----------------------------------------------------------------- */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid transparent !important;
        border-radius: 20px !important;
        background-color: white !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15) !important;
        padding: 20px !important;
        margin-bottom: 20px !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 5. 데이터 로드 및 이름 매핑
# ==============================================================================

# [설정] 모델 이름 매핑 (원래 이름 -> 보여줄 이름)
CUSTOM_NAME_MAP = {
    # ML 모델 매핑
    "lg": "로지스틱 회귀 (Logistic Regression)",
    "hgb": "Histogram-based Gradient Boosting",
    "lgbm": "LightGBM",
    
    # DL 모델 매핑
    "mlp_base": "다층 퍼셉트론 (DL1)",
    "mlp_enhance": "다층 퍼셉트론 (DL2)",
    "mlp_advanced": "다층 퍼셉트론 (DL3)"
}

# @st.cache_data
def load_model_inventory():
    """
    model_card.json을 읽고, CUSTOM_NAME_MAP에 정의된 이름으로 변환하여 로드합니다.
    """
    inventory = {"ML": {}, "DL": {}}
    
    if EVAL_ROOT and EVAL_ROOT.exists():
        for folder in EVAL_ROOT.iterdir():
            if folder.is_dir():
                card_path = folder / "model_card.json"
                if card_path.exists():
                    try:
                        with open(card_path, "r", encoding="utf-8") as f:
                            card = json.load(f)
                        
                        category = card.get("category", "ML")
                        
                        # 1. JSON에서 원래 display_name (또는 model_id) 가져오기
                        raw_name = card.get("display_name", card.get("model_id", folder.name))
                        
                        # 2. 매핑 테이블 확인해서 이름 바꿔치기
                        final_name = CUSTOM_NAME_MAP.get(raw_name.strip(), raw_name)
                        
                        if category not in inventory: 
                            inventory[category] = {}
                        
                        # 3. [중요] 바뀐 이름을 키(Key), 실제 폴더명을 값(Value)으로 저장
                        inventory[category][final_name] = folder.name
                        
                    except:
                        continue
    return inventory

# @st.cache_data # 주석처리
def load_topk_metrics(folder_name):
    if not folder_name: return None
    path = EVAL_ROOT / folder_name / "topk_metrics.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    return None

# @st.cache_data
def load_topk_cutoffs(folder_name):
    if not folder_name: return None
    path = EVAL_ROOT / folder_name / "topk_cutoffs.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    return None

def get_combined_metrics(metrics_data, cutoffs_data, k_percent):
    """
    [핵심 수정] k_percent(정수 5)와 json의 k_pct(실수 0.05 or 정수 5)를
    유연하게 비교하여 값을 찾아냅니다.
    """
    p, r, l, c = 0.0, 0.0, 0.0, 0.0
    
    # 비교를 위해 실수형(0.05)과 정수형(5) 값을 미리 준비
    k_float = k_percent / 100.0  # 0.05
    k_int = k_percent            # 5

    # 1. 지표 찾기
    if metrics_data and "metrics_by_k" in metrics_data:
        for item in metrics_data["metrics_by_k"]:
            val = item.get("k_pct")
            # 실수형 비교(0.05) 혹은 정수형 비교(5) 둘 다 허용
            if np.isclose(val, k_float) or int(val) == k_int:
                p = item.get("precision_at_k", 0)
                r = item.get("recall_at_k", 0)
                l = item.get("lift_at_k", 0)
                break
                
    # 2. 컷오프 찾기
    if cutoffs_data and "cutoffs_by_k" in cutoffs_data:
        for item in cutoffs_data["cutoffs_by_k"]:
            val = item.get("k_pct")
            if np.isclose(val, k_float) or int(val) == k_int:
                c = item.get("t_k", 0)
                break
                
    return p, r, l, c

# ==============================================================================
# 6. 메인 로직 실행
# ==============================================================================
MODEL_INVENTORY = load_model_inventory()

st.markdown("""
<div style="padding-bottom: 0px;">
    <h1 style="
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        background: linear-gradient(to right, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        padding-bottom: 5px;
    ">
        ⚖️ Model Performance Compare
    </h1>
    <p style="
        font-size: 1.1rem;
        color: #6c757d;
        margin: 0;
        font-weight: 500;
        padding-bottom: 1px;
    ">
        Top-K(상위 N%) 구간별 모델 성능 정밀 비교 대시보드
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

select, divider, _, compare = st.columns([1.5, 0.1, 0.1, 6])

# --- [왼쪽] 모델 선택 ---
with select:
    st.markdown("##### 모델 선택")
    avail_cats = [cat for cat in MODEL_INVENTORY.keys() if MODEL_INVENTORY[cat]]
    if not avail_cats:
        st.warning("⚠️ 감지된 모델이 없습니다.")
        st.info(f"참조 경로: {EVAL_ROOT}")
        st.stop()

    # Model A
    with st.container(border=True):
        st.markdown('<div style="color:#1f77b4; font-weight:bold;">🔵 Model A (Left)</div>', unsafe_allow_html=True)
        cat_a = st.radio(" ", avail_cats, key="cat_a", horizontal=True)
        models_a_map = MODEL_INVENTORY[cat_a]
        name_a = st.selectbox("Select Model", options=list(models_a_map.keys()), key="model_a")
        folder_a = models_a_map[name_a] # 실제 폴더명

    # Model B
    with st.container(border=True):
        st.markdown('<div style="color:#d62728; font-weight:bold;">🔴 Model B (Right)</div>', unsafe_allow_html=True)
        default_idx = avail_cats.index("DL") if "DL" in avail_cats else 0
        cat_b = st.radio("  ", avail_cats, key="cat_b", horizontal=True, index=default_idx)
        models_b_map = MODEL_INVENTORY[cat_b]
        name_b = st.selectbox("Select Model", options=list(models_b_map.keys()), key="model_b")
        folder_b = models_b_map[name_b] # 실제 폴더명
        
    st.markdown("</div>", unsafe_allow_html=True)

# --- [중앙] 구분선 ---
with divider:
    st.markdown('<div style="height: 700px; width: 0.1px; background-color: #d1d5db; margin: auto;"></div>', unsafe_allow_html=True)

# --- [오른쪽] 비교 및 결과 ---
with compare:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("비교할 **두 모델**을 선택하고 **전략적 Top-K(상위 N%)** 구간을 설정하세요.")
    
    # 2. 데이터 로드 (실제 폴더명 사용)
    metrics_a = load_topk_metrics(folder_a)
    cutoffs_a = load_topk_cutoffs(folder_a)
    metrics_b = load_topk_metrics(folder_b)
    cutoffs_b = load_topk_cutoffs(folder_b)
    
    # 🔥 border=True 유지 (그림자 CSS 적용됨)
    with st.container(border=True):
        st.markdown("### Target Audience & ROI Simulation")
        v, col_s1, col_s2 = st.columns([0.1 ,4, 1], gap="medium")

        with col_s1:
            k_percent = st.select_slider("🎯 Top-K 분석 범위 설정 (%)", options=[5, 10, 15, 30], value=5)
            
            # 개선된 함수로 지표 추출
            prec_a, rec_a, lift_a, cut_a = get_combined_metrics(metrics_a, cutoffs_a, k_percent)
            prec_b, rec_b, lift_b, cut_b = get_combined_metrics(metrics_b, cutoffs_b, k_percent)
            

            # 툴팁 처리
            try:
                tooltip_a_cut = model_tooltip(name_a, color='#1f77b4')
                tooltip_b_cut = model_tooltip(name_b, color='#d62728')
            except:
                tooltip_a_cut = f"<span style='color:#1f77b4'>{name_a}</span>"
                tooltip_b_cut = f"<span style='color:#d62728'>{name_b}</span>"

            st.markdown(f"""<div class='cutoff-info'>✂️ <b>Cutoff Score :</b> <span>🔵 {name_a} > <b>{cut_a:.3f}</b></span> &nbsp;|&nbsp; <span>🔴 {name_b} > <b>{cut_b:.3f}</b></span></div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

        with col_s2:
            st.metric("Target Scope", f"Top {k_percent}%", help="전체 유저 중 상위 N%")
            st.write("")
    
    col_left, col_mid_res, col_right = st.columns([1, 0.2, 1])

    # --- Model A Result ---
    with col_left:
        with st.container(border=True):
            try:
                display_a = model_tooltip(name_a, color='#1f77b4')
            except:
                display_a = f"<span style='color:#1f77b4'>{name_a}</span>"

            st.markdown(f"<div class='compare-header'><span style='font-size: 1.5rem;'>🔵</span>&nbsp;{display_a}</div>", unsafe_allow_html=True)
            
            st.info(f"Category: {cat_a}")
            
            if metrics_a:
                st.write("") 
                c1, c2, c3 = st.columns(3)
                c1.metric("Precision", f"{prec_a:.2%}", delta=f"{prec_a - prec_b:.2%}")
                c2.metric("Recall", f"{rec_a:.2%}", delta=f"{rec_a - rec_b:.2%}")
                c3.metric("Lift", f"{lift_a:.2f}x", delta=f"{lift_a - lift_b:.2f}x")
                st.write("")

                fig_a = go.Figure(data=go.Scatterpolar(
                    r=[prec_a, rec_a, min(lift_a / 5, 1.0)], 
                    theta=['Precision', 'Recall', 'Lift/5'],
                    fill='toself', name=name_a, line_color='#1f77b4'
                ))
                fig_a.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, height=250, margin=dict(t=20, b=20, l=40, r=40))
                st.plotly_chart(fig_a, use_container_width=True)
            else:
                st.warning("지표 데이터 없음")

    # --- VS Badge ---
    with col_mid_res:
        st.markdown("<div class='vs-badge-large'>VS</div>", unsafe_allow_html=True)

    # --- Model B Result ---
    with col_right:
        with st.container(border=True):
            try:
                display_b = model_tooltip(name_b, color='#d62728')
            except:
                display_b = f"<span style='color:#d62728'>{name_b}</span>"

            st.markdown(f"<div class='compare-header'><span style='font-size: 1.5rem;'>🔴</span>&nbsp;{display_b}</div>", unsafe_allow_html=True)
            
            st.error(f"Category: {cat_b}")
            
            if metrics_b:
                st.write("")
                c1, c2, c3 = st.columns(3)
                c1.metric("Precision", f"{prec_b:.2%}", delta=f"{prec_b - prec_a:.2%}")
                c2.metric("Recall", f"{rec_b:.2%}", delta=f"{rec_b - rec_a:.2%}")
                c3.metric("Lift", f"{lift_b:.2f}", delta=f"{lift_b - lift_a:.2f}")
                st.write("")

                fig_b = go.Figure(data=go.Scatterpolar(
                    r=[prec_b, rec_b, min(lift_b / 5, 1.0)], 
                    theta=['Precision', 'Recall', 'Lift/5'],
                    fill='toself', name=name_b, line_color='#d62728'
                ))
                fig_b.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, height=250, margin=dict(t=20, b=20, l=40, r=40))
                st.plotly_chart(fig_b, use_container_width=True)
            else:
                st.warning("지표 데이터 없음")