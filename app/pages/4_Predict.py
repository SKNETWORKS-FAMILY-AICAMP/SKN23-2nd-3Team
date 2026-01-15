from __future__ import annotations

from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go

from utils.paths import DEFAULT_PATHS as P, ensure_runtime_dirs
from utils.ui import apply_base_layout, hide_sidebar, top_nav

# 0) Page Config
st.set_page_config(
    page_title="AI 이탈 예측 솔루션",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

apply_base_layout()
hide_sidebar()
top_nav()

# ==== CSS 스타일링 (제공해주신 스타일 그대로 유지) =====
st.markdown("""
<style>
    .block-container { padding-top: 0.6rem !important; padding-bottom: 3rem; }
    h1 { padding-top: 0rem !important; margin-top: -2rem !important; }
    div[data-testid="stVerticalBlock"] { gap: 0.5rem !important; }
    
    .risk-badge {
        display: inline-block; padding: 0.3rem 1.2rem; border-radius: 50px;
        font-weight: 700; font-size: 0.85rem; margin-top: 0.3rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .section-header {
        color: #dd2e1f; font-size: 1.25rem; font-weight: 700;
        margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid #dd2e1f;
    }
    .stButton>button {
        width: 100%; background: linear-gradient(135deg, #dd2e1f 20%, #ffdff6 100%);
        color: white; font-weight: 700; padding: 0.9rem 2rem; border-radius: 12px;
        border: none; font-size: 1.1rem; transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(102,126,234,0.4);
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(102,126,234,0.5); }
    
    .result-wrap {
        border-radius: 18px; padding: 0.6rem; box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 0.2rem 0 1.0rem 0; border: 1px solid rgba(0,0,0,0.05);
    }
    
    /* [수정] 간격(gap)을 0.6rem -> 0.3rem으로 축소 */
    .kpi-wrap { display:flex; flex-direction: column; align-items: flex-end; gap: 0.3rem; width: 100%; }
    
    /* [수정] 카드 패딩과 글자 크기 축소 */
    .stat-card-small {
        background: linear-gradient(135deg, #dd2e1f 20%, #670800 100%);
        color: white;
        padding: 0.4rem 0.5rem; /* 패딩 축소 */
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(102,126,234,0.20);
        width: 100%; 
        margin-bottom: 0.2rem; /* 마진 축소 */
    }
    .stat-card-small .stat-label {
        font-size: 0.75rem; /* 라벨 폰트 축소 */
        font-weight: 700;
        opacity: 0.95;
        margin-bottom: 1px;
    }
    .stat-card-small .stat-value {
        font-size: 1.3rem; /* 값 폰트 축소 */
        font-weight: 900;
    }
    
    div.row-widget.stRadio > div { flex-direction: row; gap: 20px; align-items: center; }
    
    .frame-head { display:flex; justify-content: space-between; align-items: center; gap: 1rem; margin: 0.2rem 0 0.35rem 0; }
    .frame-title { display: flex; align-items: center; gap: 0.5rem; color: #dd2e1f; font-size: 1.15rem; font-weight: 700; line-height: 1.1; }
    .frame-line { height: 6px; margin: 0.1rem 0 0.4rem 0; position: relative; }
    .frame-line::before { content: ""; position: absolute; left: 0; right: 0; top: 50%; height: 2px; background: #dd2e1f; transform: translateY(-50%); border-radius: 999px; opacity: 0.95; }
    
    .kpi-pane { border-left: 4px solid #dd2e1f; padding-left: 1.0rem; height: 100%; display: flex; flex-direction: column; justify-content: flex-start; }
</style>
""", unsafe_allow_html=True)


# ================================= 제목 ==============================
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
        background: linear-gradient(135deg, #dd2e1f 20%, #ffdff6 100%);
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
        background: linear-gradient(135deg, #dd2e1f 20%, #ffdff6 100%);
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
        <h1 class="main-title">AI 고객 이탈 예측 시스템</h1>
        <p class="subtitle">딥러닝 MLP 모델을 이용한 예측</p>
        <div class="accent-line"></div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)



ensure_runtime_dirs()
sys.path.insert(0, str(P.root))


# 1) Paths / Config
DATA_PATH = P.parquet_path("features_ml_clean")
MODELS_DL_ROOT = P.models_dl_dir
PREP_ROOT = P.models_preprocessing_dir
EVAL_ROOT = P.models_eval_dir
METRICS_ROOT = P.models_metrics_dir

FEATURE_ORDER = [
    "n_events_30d", "active_days_30d", "n_purchase_30d", "purchase_ratio",
    "days_since_last_event", "days_since_last_purchase", "brand_concentration_ratio",
    "brand_switch_count_30d", "total_spend_30d", "activity_ratio_15d",
    "price_volatility", "n_events_7d", "visit_regularity", "activity_trend",
]
N_FEATURES = len(FEATURE_ORDER)

# 2) Helpers & Loaders
def read_json(path: Path):
    if not path.exists(): return None
    return json.loads(path.read_text(encoding="utf-8"))

def file_mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else -1.0

def must_exist_parquets_for_project() -> None:
    P.must_parquet_path("base")
    P.must_parquet_path("anchors")
    P.must_parquet_path("labels")
    P.must_parquet_path("features_ml_clean")

@st.cache_data
def load_feature_stats(data_path: Path, feature_order: list[str], data_mtime: float):
    df = pd.read_parquet(data_path)
    stats = {}
    for col in feature_order:
        s = df[col].dropna()
        if len(s) == 0:
            stats[col] = {"mean": 0.0, "std": 1.0}
            continue
        std = float(s.std())
        stats[col] = {"mean": float(s.mean()), "std": std if std != 0 else 1.0}
    return stats

def resolve_scaler_path(model_name: str, version: str) -> Path:
    cand = [
        PREP_ROOT / model_name / version / "scaler.pkl",
        PREP_ROOT / model_name / version / "scaler.joblib",
        PREP_ROOT / f"{model_name}_scaler.pkl",
        PREP_ROOT / f"{model_name}_scaler.joblib",
    ]
    for p in cand:
        if p.exists(): return p
    return cand[0]

def resolve_percentiles_path(model_id: str, model_name: str, version: str) -> Path:
    eval_dir = model_id.replace("__", "")
    cand = [
        METRICS_ROOT / f"{model_name}_{version}_score_percentiles.json",
        METRICS_ROOT / f"{model_name}_score_percentiles.json",
        METRICS_ROOT / f"{eval_dir}_{version}_score_percentiles.json",
        METRICS_ROOT / f"{eval_dir}_score_percentiles.json",
    ]
    for p in cand:
        if p.exists(): return p
    return cand[-1]

@st.cache_data
def load_topk_cutoffs(path: Path, path_mtime: float):
    payload = read_json(path)
    if not payload: return {}
    out = {}
    for row in payload.get("cutoffs_by_k", []):
        out[int(row["k_pct"])] = float(row["t_k"])
    if not out and isinstance(payload, dict):
        for k, v in payload.items():
            try: out[int(k)] = float(v)
            except: pass
    return out

@st.cache_data
def load_score_percentiles(path: Path, path_mtime: float):
    payload = read_json(path)
    if not payload: return None
    if isinstance(payload, list): return payload
    return payload.get("percentiles")

def percentile_label(prob: float, percentiles) -> str | None:
    if not percentiles: return None
    def get_thr(row: dict):
        for key in ("thr", "threshold", "score", "value", "t"):
            if key in row: return float(row[key])
        return None
    rows = []
    for r in percentiles:
        if not isinstance(r, dict) or "pct" not in r: continue
        thr = get_thr(r)
        if thr is None: continue
        rows.append((int(r["pct"]), float(thr)))
    if not rows: return None
    rows.sort(key=lambda x: x[0])
    for pct, thr in rows:
        if prob >= thr: return f"상위 {pct}%"
    return f"상위 {rows[-1][0]}% 밖"

def risk_from_topk(prob: float, cutoffs: dict[int, float]):
    if not cutoffs: return ("Unknown", "#888", "⚪", "#f5f5f5", None)
    ks = sorted(cutoffs.keys())
    hit_k = None
    for k in ks:
        if prob >= float(cutoffs[k]):
            hit_k = k
            break
    if hit_k == 5: return ("High Risk", "#dc3545", "🔴", "#fff5f5", hit_k)
    if hit_k == 10: return ("Medium-High", "#ff6b6b", "🟠", "#fff7f7", hit_k)
    if hit_k == 15: return ("Medium", "#ffc107", "🟡", "#fffbf0", hit_k)
    if hit_k == 30: return ("Low-Medium", "#28a745", "🟢", "#f0fff4", hit_k)
    return ("Low Risk", "#28a745", "🟢", "#f0fff4", None)

def _unwrap_state_dict(obj: object) -> dict:
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
    elif isinstance(obj, dict) and "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
        sd = obj["model_state_dict"]
    elif isinstance(obj, dict):
        sd = obj
    else: raise ValueError("invalid checkpoint format")
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

def build_dl_model(model_name: str, input_dim: int):
    from models.model_definitions import MLP_base, MLP_enhance, MLP_advanced
    name_to_ctor = {"mlp_base": MLP_base, "mlp_enhance": MLP_enhance, "mlp_advanced": MLP_advanced}
    if model_name not in name_to_ctor: raise ValueError(f"Unknown DL model_name: {model_name}")
    return name_to_ctor[model_name](input_dim=input_dim)

@st.cache_resource
def load_fixed_dl_bundle(model_name: str, version: str):
    model_dir = MODELS_DL_ROOT / model_name / version
    if not model_dir.exists(): raise FileNotFoundError(f"DL model dir not found: {model_dir}")
    cand = [model_dir / "model.pt", model_dir / "weights.pt", model_dir / f"{model_name}.pt"]
    model_path = next((p for p in cand if p.exists()), None)
    if model_path is None: raise FileNotFoundError(f"DL weights not found")
    scaler_path = resolve_scaler_path(model_name, version)
    scaler = None
    if scaler_path.exists():
        import joblib
        scaler = joblib.load(scaler_path)
    
    model = build_dl_model(model_name=model_name, input_dim=N_FEATURES)
    ckpt = torch.load(model_path, map_location="cpu")
    state = _unwrap_state_dict(ckpt)
    
    new_state = {}
    for key, value in state.items():
        new_key = key
        if ".block.0." in key: new_key = key.replace(".block.0.", ".fc1.")
        elif ".block.1." in key: new_key = key.replace(".block.1.", ".bn1.")
        elif ".block.4." in key: new_key = key.replace(".block.4.", ".fc2.")
        elif ".block.5." in key: new_key = key.replace(".block.5.", ".bn2.")
        new_state[new_key] = value
    
    try:
        model.load_state_dict(new_state, strict=True)
    except RuntimeError as e:
        model.load_state_dict(new_state, strict=False)

    model.eval()
    return model, scaler, model_path, scaler_path

def predict_prob(model, x_df: pd.DataFrame) -> float:
    x_tensor = torch.from_numpy(x_df.to_numpy(dtype=np.float32))
    with torch.no_grad():
        logits = model(x_tensor)
        prob = torch.sigmoid(logits).item()
    return float(prob)


def _parse_int_optional(label: str, raw: str, min_v: int, max_v: int):
    raw = (raw or "").strip()
    if raw == "": return 0, None  
    try:
        if '.' in raw: return 0, f"{label}: 정수만 입력해주세요."
        v = int(float(raw))
    except:
        return 0, f"{label}: 숫자를 입력해주세요."
    if v < min_v or v > max_v:
        return 0, f"{label}: {min_v:,}~{max_v:,} 사이 값이어야 합니다." 
    return v, None

def _parse_float_optional(label: str, raw: str, min_v: float, max_v: float):
    raw = (raw or "").strip()
    if raw == "": return 0.0, None 
    try:
        v = float(raw)
    except:
        return 0.0, f"{label}: 숫자를 입력해주세요."
    if v < min_v or v > max_v:
        return 0.0, f"{label}: {min_v:,}~{max_v:,} 사이 값이어야 합니다."  
    return v, None

def input_int_placeholder(label: str, key: str, min_v: int, max_v: int):
    placeholder = f"숫자 입력 ({min_v:,}~{max_v:,})"
    
    if key in st.session_state:
        raw = st.text_input(label, key=key, placeholder=placeholder)
    else:
        raw = st.text_input(label, value="", key=key, placeholder=placeholder)
    
    val, err = _parse_int_optional(label, raw, min_v, max_v)
    if key == "n_purchase_30d_txt" and val >0:
        val -= 1

    return val, err

def input_float_placeholder(label: str, key: str, min_v: float, max_v: float):
    placeholder = f"숫자 입력 ({min_v:,}~{max_v:,})"
    if key in st.session_state:
        raw = st.text_input(label, key=key, placeholder=placeholder)
    else:
        raw = st.text_input(label, value="", key=key, placeholder=placeholder)
    return _parse_float_optional(label, raw, min_v, max_v)

# 5) 파일 불러오기
must_exist_parquets_for_project()
if not DATA_PATH.exists():
    st.error(f"features file not found: {DATA_PATH}")
    st.stop()

STATS = load_feature_stats(DATA_PATH, FEATURE_ORDER, file_mtime(DATA_PATH))
FIXED_MODEL_NAME = "mlp_advanced"
FIXED_VERSION = "baseline"
MODEL_ID = "dl__mlp_advanced"
EVAL_DIR_NAME = MODEL_ID.replace("__", "")
TOPK_PATH = EVAL_ROOT / EVAL_DIR_NAME / "topk_cutoffs.json"
PCTS_PATH = resolve_percentiles_path(MODEL_ID, FIXED_MODEL_NAME, FIXED_VERSION)
topk_cutoffs = load_topk_cutoffs(TOPK_PATH, file_mtime(TOPK_PATH))
pcts = load_score_percentiles(PCTS_PATH, file_mtime(PCTS_PATH))

try:
    model, scaler, model_path, scaler_path = load_fixed_dl_bundle(FIXED_MODEL_NAME, FIXED_VERSION)
except Exception as e:
    st.error(f"모델 로드 실패: {e}"); st.stop()

# =============================================================================================================

PRESET_VALUES = {
    "Top 5%": {
        "n_events_30d_txt": "1", "active_days_30d_txt": "1", "days_since_last_event_txt": "24.375764",
        "n_purchase_30d_txt": "1", "total_spend_30d_txt": "0.0", "purchase_ratio_txt": "0.000000",
        "days_since_last_purchase_txt": "31.000000", "activity_trend_txt": "0.0",
        "brand_concentration_ratio_txt": "1.0", "brand_switch_count_30d_txt": "0",
        "visit_regularity_txt": "-1.0", "activity_ratio_15d_txt": "0.0", "n_events_7d_txt": "0", "price_volatility_txt": "0.0"
    },
    "Top 15%": {
        "n_events_30d_txt": "1", "active_days_30d_txt": "1", "n_purchase_30d_txt": "1",
        "purchase_ratio_txt": "0.000000", "days_since_last_event_txt": "17.657164",
        "days_since_last_purchase_txt": "31.00000", "brand_concentration_ratio_txt": "1.0",
        "brand_switch_count_30d_txt": "0", "total_spend_30d_txt": "0.0",
        "activity_ratio_15d_txt": "0.0", "n_events_7d_txt": "0",
        "visit_regularity_txt": "-1.0", "activity_trend_txt": "0.0", "price_volatility_txt": "0.0"
    },
    "Top 30%": {
        "n_events_30d_txt": "6",
        "active_days_30d_txt": "1",
        "n_purchase_30d_txt": "1",
        "purchase_ratio_txt": "0.05",
        "days_since_last_event_txt": "26.304306",
        "days_since_last_purchase_txt": "26.312245",
        "brand_concentration_ratio_txt": "1.0",
        "brand_switch_count_30d_txt": "0",
        "total_spend_30d_txt": "295.73",
        "activity_ratio_15d_txt": "0.0",
        "n_events_7d_txt": "0",
        "visit_regularity_txt": "0.00041",
        "activity_trend_txt": "0.0",
        "price_volatility_txt": "0.0"
    }
    # "Top 30%": {
    #    "n_events_30d_txt": "4", "active_days_30d_txt": "1", "n_purchase_30d_txt": "1",
    #     "purchase_ratio_txt": "0.000000", "days_since_last_event_txt": "20.731053",
    #     "days_since_last_purchase_txt": "31.00000", "brand_concentration_ratio_txt": "1.0",
    #     "brand_switch_count_30d_txt": "0", "total_spend_30d_txt": "0.0",
    #     "activity_ratio_15d_txt": "0.0", "n_events_7d_txt": "0",
    #     "visit_regularity_txt": "0.000193", "activity_trend_txt": "0.0", "price_volatility_txt": "0.0"
    # }
}

# [수정] load_preset: 값만 채우고 UI 리프레시 (예측 로직 X)
def load_preset(preset_name):
    if preset_name in PRESET_VALUES:
        for key, val in PRESET_VALUES[preset_name].items():
            st.session_state[key] = val
        st.rerun()

col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    # 1. 탭 스타일 커스텀 CSS 주입
    st.markdown("""
    <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px; 
            border-bottom: 2px solid #dd2e1f !important; 
            padding-bottom: 4px !important; 
        }
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: transparent !important;
            height: 0px !important;
        }
        .stTabs [data-baseweb="tab"] {
            padding-top: 0px !important;
            padding-bottom: 0px !important;
            margin-bottom: 0px !important; 
            margin-top: 0px !important;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: #dd2e1f !important;
            font-weight: 900 !important;
            border-bottom: 6px solid #dd2e1f !important; 
        }
    </style>
    """, unsafe_allow_html=True)

    # 2. 탭 생성
    tab_5, tab_15, tab_30 = st.tabs(["🔴 Top 5%", "🟠 Top 15%", "🟢 Top 30%"])

    with tab_5:
        if st.button("이 데이터 적용하기", key="btn_top5", use_container_width=True):
            load_preset("Top 5%")

    with tab_15:
        if st.button("이 데이터 적용하기", key="btn_top15", use_container_width=True):
            load_preset("Top 15%")

    with tab_30:
        if st.button("이 데이터 적용하기", key="btn_top30", use_container_width=True):
            load_preset("Top 30%")

    # [수정] 들여쓰기 오류 해결 (with col_left 안으로 이동)
    st.markdown('<div style="border-bottom: 2px solid #dd2e1f; margin-bottom: 1rem; margin-top: 0.5rem;"></div>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown('<div class="section-header">📊 핵심 활동 지표</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            active_days_30d, err_active_days_30d = input_int_placeholder("30일 활동일수", key="active_days_30d_txt", min_v=0, max_v=30)
            n_events_30d, err_n_events_30d = input_int_placeholder("30일 이벤트 수", key="n_events_30d_txt", min_v=0, max_v=1824)
            n_purchase_30d, err_n_purchase_30d = input_int_placeholder("최근 30일 구매 횟수", key="n_purchase_30d_txt", min_v=0, max_v=512)
        with c2:
            days_since_last_event, err_days_since_last_event = input_float_placeholder("마지막 활동 (일)", key="days_since_last_event_txt", min_v=0, max_v=31)
            days_since_last_purchase, err_days_since_last_purchase = input_float_placeholder("마지막 구매 (일)", key="days_since_last_purchase_txt", min_v=0, max_v=31)
            purchase_ratio, err_purchase_ratio = input_float_placeholder("구매 전환율 (0~1)", key="purchase_ratio_txt", min_v=0.0, max_v=1.0)
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("🔧 고급 분석 지표", expanded=False):
            st.markdown('<div class="section-header">🔎 활동 추세</div>', unsafe_allow_html=True)
            activity_trend, err_activity_trend = input_float_placeholder("활동 추세", key="activity_trend_txt", min_v=0.0, max_v=1.0)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="section-header">🏷️ 브랜드 패턴</div>', unsafe_allow_html=True)
            cc1, cc2 = st.columns(2)
            with cc1:
                brand_concentration_ratio, err_brand_concentration_ratio = input_float_placeholder("브랜드 집중도 (0~1)", key="brand_concentration_ratio_txt", min_v=0.0, max_v=1.0)
                brand_switch_count_30d, err_brand_switch_count_30d = input_int_placeholder("브랜드 전환 횟수", key="brand_switch_count_30d_txt", min_v=0, max_v=616)
            with cc2:
                visit_regularity, err_visit_regularity = input_float_placeholder("방문 규칙성 (-1~21)", key="visit_regularity_txt", min_v=-1.0, max_v=21.0)
                activity_ratio_15d, err_activity_ratio_15d = input_float_placeholder("15일 활동 비중 (0~1)", key="activity_ratio_15d_txt", min_v=0, max_v=1.0)
            n_events_7d, err_n_events_7d = input_int_placeholder("7일 이벤트 수", key="n_events_7d_txt", min_v=0, max_v=311)
            price_volatility, err_price_volatility = input_float_placeholder("가격 변동성 (0~553)", key="price_volatility_txt", min_v=0, max_v=553.0)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # [중요] submit 버튼 변수 할당
        submit = st.form_submit_button("예측하기", use_container_width=True)


# 예측 및 결과

def collect_errors(*errs):
    return [e for e in errs if e is not None]

# 초기화
radar_values = [0.0, 0.0, 0.0, 0.0, 0.0]
prob = 0.0
latency_ms = 0.0
risk_level = "Ready"
risk_color = "#e9ecef"
risk_icon = "⚪"
risk_bg = "#f8f9fa"
hit_k = None
pct_label = ""
is_analyzed = False
user_inputs = {}

# [핵심] submit(예측하기 버튼)을 눌렀을 때만 실행
if submit:
    # 1. 에러 체크
    errors = collect_errors(
        err_n_purchase_30d, err_days_since_last_event, err_n_events_30d, err_active_days_30d, err_activity_trend,
        # err_total_spend_30d, # 주석 처리됨
        err_days_since_last_purchase, err_purchase_ratio,
        err_brand_concentration_ratio, err_brand_switch_count_30d, err_visit_regularity, 
        err_activity_ratio_15d, err_n_events_7d, err_price_volatility
    )

    if errors:
        st.error("입력값을 확인해주세요: " + ", ".join(errors))
    else:
        # 2. [중요] UI에서 주석 처리한 변수는 여기서 기본값을 0.0으로 만들어줘야 에러가 안 납니다.
        total_spend_30d = 0.0 

        # 3. 입력 딕셔너리 생성 (들여쓰기 정렬 완료)
        user_inputs = {
            "n_events_30d": int(n_events_30d), 
            "active_days_30d": int(active_days_30d), 
            "n_purchase_30d": int(n_purchase_30d),
            "purchase_ratio": float(purchase_ratio), 
            "days_since_last_event": float(days_since_last_event),
            "days_since_last_purchase": float(days_since_last_purchase), 
            "brand_concentration_ratio": float(brand_concentration_ratio),
            "brand_switch_count_30d": int(brand_switch_count_30d), 
            "total_spend_30d": float(total_spend_30d), # 위에서 0.0으로 정의함
            "activity_ratio_15d": float(activity_ratio_15d), 
            "price_volatility": float(price_volatility),
            "n_events_7d": int(n_events_7d), 
            "visit_regularity": float(visit_regularity), 
            "activity_trend": float(activity_trend),
        }

        x_df = pd.DataFrame([[user_inputs[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)

        with st.spinner("⚡ AI 모델 분석 중..."):
            t0 = time.time()
            if scaler is not None:
                x_scaled = scaler.transform(x_df)
                x_in = pd.DataFrame(x_scaled, columns=FEATURE_ORDER)
            else:
                arr = []
                for col in FEATURE_ORDER:
                    mean_v = float(STATS[col]["mean"]); std_v = float(STATS[col]["std"])
                    raw = float(user_inputs[col])
                    arr.append(0.0 if std_v == 0 else (raw - mean_v) / std_v)
                x_in = pd.DataFrame([arr], columns=FEATURE_ORDER)

            prob = predict_prob(model, x_in)
            latency_ms = (time.time() - t0) * 1000
            
            risk_level, risk_color, risk_icon, risk_bg, hit_k = risk_from_topk(prob, topk_cutoffs)
            pct_label = percentile_label(prob, pcts)
            
            val_activity_freq = 0.0
            if user_inputs["n_events_30d"] > 0:
                val_activity_freq = 1.0 - min(user_inputs["n_events_30d"] / 1000, 1.0)

            val_spend_score = 0.0
            if user_inputs["total_spend_30d"] > 0:
                val_spend_score = 1.0 - min(user_inputs["total_spend_30d"] / 1_000_000, 1.0)

            val_ratio_score = 0.0
            if user_inputs["purchase_ratio"] > 0:
                val_ratio_score = 1.0 - float(user_inputs["purchase_ratio"])
            
            radar_values = [
                min(user_inputs["days_since_last_event"] / 60, 1.0),
                val_activity_freq,
                val_spend_score,
                val_ratio_score,
                min(abs(user_inputs["activity_trend"]) / 10, 1.0) if user_inputs["activity_trend"] < 0 else 0
            ]
            
            is_analyzed = True

# 시각화 부분 
categories = ["최근성", "활동빈도", "전환율", "활동추세"]
# 레이더 차트 값도 차원에 맞춰 조정 (구매액 제외 4개로 줄이신 경우)
radar_values_4dim = [
    radar_values[0], # 최근성
    radar_values[1], # 활동빈도
    radar_values[3], # 전환율 (구매액 인덱스 2 건너뜀)
    radar_values[4]  # 활동추세
]

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(
    r=radar_values_4dim, 
    theta=categories, 
    fill='toself',
    fillcolor=f'rgba({int(risk_color[1:3], 16)}, {int(risk_color[3:5], 16)}, {int(risk_color[5:7], 16)}, 0.25)' if is_analyzed else 'rgba(200,200,200,0.2)',
    line=dict(color=risk_color if is_analyzed else '#ccc', width=3),
    marker=dict(size=10, color=risk_color if is_analyzed else '#ccc'),
    hovertemplate='%{theta}: %{r:.1%}<extra></extra>', name='위험도'
))

# [수정] 레이더 차트 높이 축소 반영 (height=220)
fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1], tickformat='.0%', gridcolor='#e8e8e8', tickfont=dict(size=10)),
        angularaxis=dict(gridcolor='#e8e8e8', tickfont=dict(size=11, color='#333'))
    ),
    showlegend=False, 
    height=220,  # <-- 높이 220으로 축소
    margin=dict(l=35, r=35, t=20, b=20),
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
)

with col_right:
    # margin: 상단0 좌우0 하단6px (단위를 px나 rem 중 원하는 것으로 변경하세요)
    st.markdown('<div class="section-header" style="padding-bottom: 0px;margin: 0.95rem 0px 16px;">🎯 예측 결과</div>', unsafe_allow_html=True)
    latency_display = f"⚡ {latency_ms:.2f} ms" if is_analyzed else "Ready"
    latency_txt_color = "#28a745" if is_analyzed else "#ccc"
    sub = []
    if hit_k is not None: sub.append(f"Top {hit_k}%")
    if pct_label: sub.append(pct_label)
    sub_txt = " | ".join(sub) if sub else "데이터를 입력하고 예측 버튼을 눌러주세요"

    display_prob = (prob - 0.01) * 100 if is_analyzed else 0.0
    if display_prob < 0: display_prob = 0.0

    st.markdown(f"""
        <div class="result-wrap" style="background:{risk_bg};">
            <div class="result-card" style="background: transparent; box-shadow:none; margin:0; padding: 0.2rem;">
                <div style="text-align: center;">
                    <div style="font-size: 0.8rem; color: #666; margin-bottom: 0.2rem;">{sub_txt}</div>
                    <div style="color: {risk_color if is_analyzed else '#ccc'}; font-size: 2.4rem; font-weight: 800; line-height: 1.1; margin: 0;">
                        {display_prob:.1f}%
                    </div>
                    <div style="font-size: 0.85rem; color: #888; margin-bottom: 0.5rem;">이탈 확률</div>
                    <div class="risk-badge" style="background: {risk_color}; color: {'white' if is_analyzed else '#666'}; padding: 0.3rem 1.2rem; font-size: 0.9rem; margin-top: 0;">
                        {risk_icon} {risk_level}
                    </div>
                    <div style="margin-top: 0.4rem; font-size: 0.75rem; color: {latency_txt_color}; font-weight: 600;">{latency_display}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    t1, t2 = st.columns([2,1])
    with t1:
        # [수정] style="margin-top: 5px; margin-bottom: 5px;" 추가
        st.markdown("""
            <div class="frame-head" style="margin-top: 5px; margin-bottom: 5px;">
                <div class="frame-title">📊 위험 요인 분석</div>
            </div>
            <div class="frame-line" style="margin-top: 5px; margin-bottom: 30px;"></div>
            """, unsafe_allow_html=True)
        st.plotly_chart(fig_radar, use_container_width=True)

    with t2:
        # [수정] style="margin-top: 5px; margin-bottom: 5px;" 추가
        st.markdown("""
            <div class="frame-head" style="margin-top: 5px; margin-bottom: 5px;">
                <div class="frame-title">📈 핵심 지표</div>
            </div>
            <div class="frame-line" style="margin-top: 5px; margin-bottom: 20px;"></div>
            """, unsafe_allow_html=True)
        st.markdown('<div class="kpi-pane">', unsafe_allow_html=True)
        
        # [수정] user_inputs 안전하게 접근하여 KPI 표시
        if is_analyzed:
            val_days = f"{int(user_inputs['days_since_last_event'])}일"
            val_spend = f"{user_inputs['total_spend_30d']/10000:.0f}만원"
            val_ratio = f"{user_inputs['purchase_ratio']*100:.1f}%"
        else:
            val_days = "-"
            val_spend = "-"
            val_ratio = "-"

        st.markdown(f"""
            <div class="kpi-wrap">
                <div class="stat-card-small"><div class="stat-label">최근 활동</div><div class="stat-value">{val_days}</div></div>
                <div class="stat-card-small"><div class="stat-label">전환율</div><div class="stat-value">{val_ratio}</div></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)