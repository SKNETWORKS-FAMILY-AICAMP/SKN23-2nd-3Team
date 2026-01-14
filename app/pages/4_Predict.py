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

# 0) Page Config + Refined CSS
st.set_page_config(
    page_title="AI 이탈 예측 솔루션",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

apply_base_layout()
hide_sidebar()
top_nav()

# ==== 간격 조정 =====
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

# =============================
# CSS 스타일링 (KPI 카드 디자인 복구)
# =============================
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700&display=swap');
    * { font-family: 'Noto Sans KR', sans-serif; }

    /* 결과 카드 스타일 */
    .result-card {
        background: white;
        padding: 0.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.12);
        margin: 0;
    }

    /* 결과 래퍼 스타일 */
    .result-wrap {
        border-radius: 18px;
        padding: 0.6rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 0.2rem 0 1.0rem 0;
        border: 1px solid rgba(0,0,0,0.05);
    }

    /* 뱃지 스타일 */
    .risk-badge {
        display: inline-block;
        padding: 0.3rem 1.2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 0.85rem;
        margin-top: 0.3rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    .section-header {
        color: #667eea;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }

    /* 버튼 스타일 */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 700;
        padding: 0.9rem 2rem;
        border-radius: 12px;
        border: none;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(102,126,234,0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102,126,234,0.5);
    }

    /* 입력창 스타일 */
    .stTextInput input {
        border-radius: 8px !important;
        border: 2px solid #e8e8e8 !important;
        padding: 0.55rem !important;
        background: white !important;
        transition: all 0.2s ease;
    }
    .stTextInput input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102,126,234,0.15) !important;
    }

    .input-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }

    /* KPI 카드 스타일 (작은 버전) */
    .stat-card-small {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.75rem 0.6rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(102,126,234,0.20);
        width: 100%; /* 부모 컨테이너에 맞춤 */
        margin-bottom: 0.6rem;
    }
    .stat-card-small .stat-label {
        font-size: 0.82rem;
        font-weight: 700;
        opacity: 0.95;
        margin-bottom: 2px;
    }
    .stat-card-small .stat-value {
        font-size: 1.5rem; /* 폰트 사이즈 조정 */
        font-weight: 900;
    }

    /* 프레임 헤더 */
    .frame-head {
        display:flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        margin: 0.2rem 0 0.35rem 0;
    }
    .frame-title {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #667eea;
        font-size: 1.15rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .frame-line {
        height: 6px;
        margin: 0.1rem 0 0.4rem 0;
        position: relative;
    }
    .frame-line::before {
        content: "";
        position: absolute;
        left: 0; right: 0;
        top: 50%;
        height: 2px;
        background: #667eea;
        transform: translateY(-50%);
        border-radius: 999px;
        opacity: 0.95;
    }

    /* KPI 패널 */
    .kpi-pane {
        border-left: 4px solid #667eea;
        padding-left: 1.0rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
    }
    
    /* 라디오 버튼 가로 정렬 커스텀 */
    div.row-widget.stRadio > div { 
        flex-direction: row; 
        gap: 20px; 
        align-items: center; 
    }
</style>
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
    "n_events_30d",
    "active_days_30d",
    "n_purchase_30d",
    "purchase_ratio",
    "days_since_last_event",
    "days_since_last_purchase",
    "brand_concentration_ratio",
    "brand_switch_count_30d",
    "total_spend_30d",
    "activity_ratio_15d",
    "price_volatility",
    "n_events_7d",
    "visit_regularity",
    "activity_trend",
]
N_FEATURES = len(FEATURE_ORDER)


# 2) Helpers

def read_json(path: Path):
    if not path.exists():
        return None
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
        if p.exists():
            return p
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
        if p.exists():
            return p
    return cand[-1]

@st.cache_data
def load_topk_cutoffs(path: Path, path_mtime: float):
    payload = read_json(path)
    if not payload:
        return {}
    out = {}
    for row in payload.get("cutoffs_by_k", []):
        out[int(row["k_pct"])] = float(row["t_k"])
    if not out and isinstance(payload, dict):
        for k, v in payload.items():
            try:
                out[int(k)] = float(v)
            except Exception:
                pass
    return out

@st.cache_data
def load_score_percentiles(path: Path, path_mtime: float):
    payload = read_json(path)
    if not payload:
        return None
    if isinstance(payload, list):
        return payload
    return payload.get("percentiles")

def percentile_label(prob: float, percentiles) -> str | None:
    if not percentiles:
        return None

    def get_thr(row: dict):
        for key in ("thr", "threshold", "score", "value", "t"):
            if key in row:
                return float(row[key])
        return None

    rows = []
    for r in percentiles:
        if not isinstance(r, dict) or "pct" not in r:
            continue
        thr = get_thr(r)
        if thr is None:
            continue
        rows.append((int(r["pct"]), float(thr)))

    if not rows:
        return None

    rows.sort(key=lambda x: x[0])

    for pct, thr in rows:
        if prob >= thr:
            return f"상위 {pct}%"
    return f"상위 {rows[-1][0]}% 밖"

def risk_from_topk(prob: float, cutoffs: dict[int, float]):
    if not cutoffs:
        return ("Unknown", "#888", "⚪", "#f5f5f5", None)

    ks = sorted(cutoffs.keys())
    hit_k = None
    for k in ks:
        if prob >= float(cutoffs[k]):
            hit_k = k
            break

    if hit_k == 5:
        return ("High Risk", "#dc3545", "🔴", "#fff5f5", hit_k)
    if hit_k == 10:
        return ("Medium-High", "#ff6b6b", "🟠", "#fff7f7", hit_k)
    if hit_k == 15:
        return ("Medium", "#ffc107", "🟡", "#fffbf0", hit_k)
    if hit_k == 30:
        return ("Low-Medium", "#28a745", "🟢", "#f0fff4", hit_k)

    return ("Low Risk", "#28a745", "🟢", "#f0fff4", None)

def _unwrap_state_dict(obj: object) -> dict:
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
    elif isinstance(obj, dict) and "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
        sd = obj["model_state_dict"]
    elif isinstance(obj, dict):
        sd = obj
    else:
        raise ValueError("invalid checkpoint format")

    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

def build_dl_model(model_name: str, input_dim: int):
    from models.model_definitions import MLP_base, MLP_enhance, MLP_advanced
    name_to_ctor = {
        "mlp_base": MLP_base,
        "mlp_enhance": MLP_enhance,
        "mlp_advanced": MLP_advanced,
    }
    if model_name not in name_to_ctor:
        raise ValueError(f"Unknown DL model_name: {model_name}")
    return name_to_ctor[model_name](input_dim=input_dim)

@st.cache_resource
def load_fixed_dl_bundle(model_name: str, version: str):
    model_dir = MODELS_DL_ROOT / model_name / version
    if not model_dir.exists():
        raise FileNotFoundError(f"DL model dir not found: {model_dir}")

    cand = [
        model_dir / "model.pt",
        model_dir / "weights.pt",
        model_dir / f"{model_name}.pt",
    ]
    model_path = next((p for p in cand if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError(
            f"DL weights not found under: {model_dir} "
            f"(expected model.pt / weights.pt / {model_name}.pt)"
        )

    scaler_path = resolve_scaler_path(model_name, version)
    scaler = None
    if scaler_path.exists():
        import joblib
        scaler = joblib.load(scaler_path)

    model = build_dl_model(model_name=model_name, input_dim=N_FEATURES)
    ckpt = torch.load(model_path, map_location="cpu")
    state = _unwrap_state_dict(ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    return model, scaler, model_path, scaler_path

def predict_prob(model, x_df: pd.DataFrame) -> float:
    x_tensor = torch.from_numpy(x_df.to_numpy(dtype=np.float32))
    with torch.no_grad():
        logits = model(x_tensor)
        prob = torch.sigmoid(logits).item()
    return float(prob)


# ==============================================================================
# 3) Placeholder 기반 숫자 입력 헬퍼 (수정됨: 문자 및 빈값 엄격 체크)
# ==============================================================================

def _parse_int_optional(label: str, raw: str, min_v: int, max_v: int):
    raw = (raw or "").strip()
    
    # 1. 빈 값은 0으로 처리 (에러 아님) -> 사용자 요청 반영
    if raw == "":
        return 0, None
    
    try:
        # 2. 숫자 변환 및 정수 체크
        if '.' in raw:
             return 0, f"{label}: 정수(소수점 없음)만 입력해주세요."
        
        v = int(float(raw)) 
    except Exception:
        # 문자가 섞여있으면 여기서 걸림
        return 0, f"{label}: 올바른 숫자 형식이 아닙니다."
        
    # 3. 범위 체크
    if v < min_v or v > max_v:
        return 0, f"{label}: {min_v} ~ {max_v} 사이의 값이어야 합니다."
        
    return v, None

def _parse_float_optional(label: str, raw: str, min_v: float, max_v: float):
    raw = (raw or "").strip()
    
    # 1. 빈 값은 0으로 처리 (에러 아님)
    if raw == "":
        return 0.0, None
    
    try:
        # 2. 숫자 변환 시도
        v = float(raw)
    except Exception:
        # 문자가 섞여있으면 여기서 걸림
        return 0.0, f"{label}: 올바른 숫자 형식이 아닙니다."
        
    # 3. 범위 체크
    if v < min_v or v > max_v:
        return 0.0, f"{label}: {min_v} ~ {max_v} 사이의 값이어야 합니다."
        
    return v, None

# ✅ 세션 상태(st.session_state)의 값을 우선적으로 사용하여 입력창에 반영
def input_int_placeholder(label: str, key: str, min_v: int, max_v: int):
    placeholder = f"숫자 입력 ({min_v}~{max_v})"
    default_val = st.session_state.get(key, "")
    raw = st.text_input(label, value=default_val, key=key, placeholder=placeholder)
    return _parse_int_optional(label, raw, min_v, max_v)

def input_float_placeholder(label: str, key: str, min_v: float, max_v: float):
    placeholder = f"숫자 입력 ({min_v}~{max_v})"
    default_val = st.session_state.get(key, "")
    raw = st.text_input(label, value=default_val, key=key, placeholder=placeholder)
    return _parse_float_optional(label, raw, min_v, max_v)


# 4) 팀 정의(없음=0) 기반 연관 규칙 적용

def apply_team_rules(x: dict) -> dict:
    x = dict(x)

    # ---- 구매 블록 ----
    if x["n_purchase_30d"] == 0:
        x["total_spend_30d"] = 0.0
        x["purchase_ratio"] = 0.0
        x["days_since_last_purchase"] = 0

    if x["total_spend_30d"] == 0:
        x["n_purchase_30d"] = 0
        x["purchase_ratio"] = 0.0
        x["days_since_last_purchase"] = 0

    # ---- 활동 블록 ----
    if x["active_days_30d"] == 0:
        x["days_since_last_event"] = 0
        x["n_events_30d"] = 0
        x["n_events_7d"] = 0
        x["activity_ratio_15d"] = 0.0
        x["visit_regularity"] = 0.0
        x["activity_trend"] = 0.0
        x["price_volatility"] = 0.0

        x["brand_concentration_ratio"] = 0.0
        x["brand_switch_count_30d"] = 0

        x["n_purchase_30d"] = 0
        x["total_spend_30d"] = 0.0
        x["purchase_ratio"] = 0.0
        x["days_since_last_purchase"] = 0

    if x["n_events_30d"] == 0:
        x["active_days_30d"] = 0
        x["days_since_last_event"] = 0
        x["n_events_7d"] = 0
        x["activity_ratio_15d"] = 0.0
        x["visit_regularity"] = 0.0
        x["activity_trend"] = 0.0
        x["price_volatility"] = 0.0
        x["brand_concentration_ratio"] = 0.0
        x["brand_switch_count_30d"] = 0

    if x["n_events_7d"] > x["n_events_30d"]:
        x["n_events_7d"] = x["n_events_30d"]

    return x


# 5) Load required files + Fixed model resources

must_exist_parquets_for_project()

if not DATA_PATH.exists():
    st.error(f"features file not found: {DATA_PATH}")
    st.stop()

STATS = load_feature_stats(DATA_PATH, FEATURE_ORDER, file_mtime(DATA_PATH))

FIXED_MODEL_NAME = "mlp_enhance"
FIXED_VERSION = "baseline"
MODEL_ID = "dl__mlp_enhance"
EVAL_DIR_NAME = MODEL_ID.replace("__", "")

TOPK_PATH = EVAL_ROOT / EVAL_DIR_NAME / "topk_cutoffs.json"
PCTS_PATH = resolve_percentiles_path(MODEL_ID, FIXED_MODEL_NAME, FIXED_VERSION)

topk_cutoffs = load_topk_cutoffs(TOPK_PATH, file_mtime(TOPK_PATH))
pcts = load_score_percentiles(PCTS_PATH, file_mtime(PCTS_PATH))

try:
    model, scaler, model_path, scaler_path = load_fixed_dl_bundle(FIXED_MODEL_NAME, FIXED_VERSION)
except Exception as e:
    st.error(f"모델 로드 실패: {e}")
    st.stop()


# 6) UI - Header

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
        ⚡ AI 고객 이탈 예측 시스템
    </h1>
    <p style="
        font-size: 1.1rem;
        color: #6c757d;
        margin: 0;
        font-weight: 500;
        padding-bottom: 25px;
    ">
        딥러닝 MLP 모델을 이용한 예측 
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# --- [Preset] 시뮬레이션 데이터 정의 ---
PRESET_VALUES = {
    "🔴 이탈위험 높음": {
        "n_evt_txt": "0", "act_days_txt": "0", "days_evt_txt": "0",
        "n_purchase_control_txt": "0", "spend_txt": "0", "purchase_ratio_txt": "0", "days_pur_txt": "0",
        "trend_txt": "0", "conc_txt": "0", "switch_txt": "0",
        "reg_txt": "0", "ratio15_txt": "0", "evt7_txt": "0", "vol_txt": "0"
    },
    "🟡 보통": {
        "n_evt_txt": "60", "act_days_txt": "3", "days_evt_txt": "14",
        "n_purchase_control_txt": "1", "spend_txt": "45000", "purchase_ratio_txt": "0.02", "days_pur_txt": "20",
        "trend_txt": "-2.5", "conc_txt": "0.5", "switch_txt": "2",
        "reg_txt": "0.25", "ratio15_txt": "0.25", "evt7_txt": "3", "vol_txt": "2.0"
    },
    "🟢 매우 낮음": {
        "n_evt_txt": "4200", "act_days_txt": "18", "days_evt_txt": "1",
        "n_purchase_control_txt": "6", "spend_txt": "980000", "purchase_ratio_txt": "0.18", "days_pur_txt": "2",
        "trend_txt": "3.2", "conc_txt": "0.8", "switch_txt": "5",
        "reg_txt": "0.78", "ratio15_txt": "0.62", "evt7_txt": "950", "vol_txt": "5.0"
    }
}

def update_inputs_from_preset():
    selected = st.session_state.get("risk_preset_radio")
    if selected in PRESET_VALUES:
        for key, val in PRESET_VALUES[selected].items():
            st.session_state[key] = val

# 7) UI - Input Form (좌:우 = 3:2)

col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    # 🌟 라디오 버튼 (시뮬레이션 프리셋)
    # st.markdown("##### ⚡ 시뮬레이션 설정")
    # st.radio(
    #     "이탈 위험도 시나리오 선택",
    #     options=["🔴 매우 높음", "🟡 보통", "🟢 매우 낮음"],
    #     key="risk_preset_radio",
    #     horizontal=True,
    #     index=None, # 초기 선택 없음
    #     on_change=update_inputs_from_preset
    # )
    

    # 🌟 [수정] 헤더와 라디오 버튼을 가로로 나란히 배치 및 정렬
    # vertical_alignment="center": 텍스트와 버튼의 높이 중심을 맞춤
    d1, d2, d3 = st.columns([1, 0.5, 1], vertical_alignment="center")
    
    with d1:
        # style="margin-bottom:0; border:none;" 추가하여 불필요한 여백과 밑줄 제거 (깔끔하게 보이기 위함)
        st.markdown('<div class="section-header" style="margin-bottom: 1; padding-bottom: 0; border-bottom: none;">🛍️ 구매 활동</div>', unsafe_allow_html=True)
    
    with d3:
        st.radio(
            "이탈위험 프리셋", # 접근성을 위해 라벨 텍스트는 작성
            options=["매우 높음", "보통", "매우 낮음"],
            key="risk_preset_radio",
            horizontal=True,
            index=None,
            on_change=update_inputs_from_preset,
            label_visibility="collapsed" # 🔥 핵심: 라벨 공간을 완전히 제거하여 위로 올림
        )
    
    # 두 요소 아래에 공통 구분선 추가 (선택 사항)
    st.markdown('<div style="border-bottom: 2px solid #667eea; margin-bottom: 1rem; margin-top: -1rem;"></div>', unsafe_allow_html=True)

    # [중요] 여기서도 동일한 검증 함수(input_int_placeholder)를 사용하므로
    # 빈 값, 문자, 범위 오류 시 err_n_pur에 에러 메시지가 담깁니다.
    n_pur, err_n_pur = input_int_placeholder(
        "최근 30일 구매 횟수",
        key="n_purchase_control_txt",
        min_v=0,
        max_v=500,
    )

    st.markdown("</div>", unsafe_allow_html=True)


    with st.form("prediction_form"):
        st.markdown('<div class="section-header">📊 핵심 활동 지표</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            days_evt, err_days_evt = input_int_placeholder(
                "📅 마지막 활동 (일)",
                key="days_evt_txt",
                min_v=-30,
                max_v=30,
            )
            n_evt, err_n_evt = input_int_placeholder(
                "🔄 30일 이벤트 수",
                key="n_evt_txt",
                min_v=0,
                max_v=20000,
            )

        with c2:
            act_days, err_act_days = input_int_placeholder(
                "📆 30일 활동일수",
                key="act_days_txt",
                min_v=0,
                max_v=30,
            )
            trend, err_trend = input_float_placeholder(
                "📈 활동 추세",
                key="trend_txt",
                min_v=-10.0,
                max_v=10.0,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("🔧 고급 분석 지표", expanded=False):
            st.markdown('<div class="section-header">💰 구매 행동</div>', unsafe_allow_html=True)

            c3, c4 = st.columns(2)
            with c3:
                spend, err_spend = input_float_placeholder(
                    "총 구매액 (원)",
                    key="spend_txt",
                    min_v=0.0,
                    max_v=10_000_000.0,
                )
            with c4:
                days_pur, err_days_pur = input_int_placeholder(
                    "📦 마지막 구매 (일)",
                    key="days_pur_txt",
                    min_v=0,
                    max_v=365,
                )

            purchase_ratio, err_purchase_ratio = input_float_placeholder(
                "🛒 구매 전환율 (0~1)",
                key="purchase_ratio_txt",
                min_v=0.0,
                max_v=1.0,
            )

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">🏷️ 브랜드 패턴</div>', unsafe_allow_html=True)

            cc1, cc2 = st.columns(2)
            with cc1:
                conc, err_conc = input_float_placeholder(
                    "브랜드 집중도 (0~1)",
                    key="conc_txt",
                    min_v=0.0,
                    max_v=1.0,
                )
                switch, err_switch = input_int_placeholder(
                    "브랜드 전환 횟수",
                    key="switch_txt",
                    min_v=0,
                    max_v=200,
                )
            with cc2:
                reg, err_reg = input_float_placeholder(
                    "방문 규칙성 (0~1)",
                    key="reg_txt",
                    min_v=0.0,
                    max_v=1.0,
                )
                ratio15, err_ratio15 = input_float_placeholder(
                    "15일 활동 비중 (0~1)",
                    key="ratio15_txt",
                    min_v=0.0,
                    max_v=1.0,
                )

            evt7, err_evt7 = input_int_placeholder(
                "7일 이벤트 수",
                key="evt7_txt",
                min_v=0,
                max_v=10000,
            )

            vol, err_vol = input_float_placeholder(
                "💸 가격 변동성 (0~10)",
                key="vol_txt",
                min_v=0.0,
                max_v=10.0,
            )

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("이탈 위험 예측하기", use_container_width=True)


# ==============================================================================
# 8) Predict + Result UI
# ==============================================================================

def collect_errors(*errs):
    return [e for e in errs if e is not None]

# 1. [초기화] 변수 설정
radar_values = [0.0, 0.0, 0.0, 0.0, 0.0]
prob = 0.0
latency_ms = 0.0
risk_level = "Ready"
risk_color = "#e9ecef"
risk_icon = "⚪"
risk_bg = "#f8f9fa"
hit_k = None
pct_label = ""
is_analyzed = False # 분석 완료 플래그

# 2. 버튼 클릭 시 로직
if submit:
    # 에러 체크 (구매 횟수 포함)
    errors = collect_errors(
        err_n_pur, # 구매 횟수 에러
        err_days_evt, err_n_evt, err_act_days, err_trend,
        err_spend, err_days_pur, err_purchase_ratio,
        err_conc, err_switch, err_reg, err_ratio15,
        err_evt7, err_vol
    )

    # [수정] 에러가 하나라도 있으면 경고 메시지 띄우고 중단 (실행 차단)
    if errors:
        st.toast("⚠️ 잘못된 값이 입력되었습니다. 숫자만 입력해주세요.", icon="🚫")
        st.stop() # 여기서 코드 실행을 멈춤

    # 입력값 정리
    user_inputs = {
        "n_events_30d": int(n_evt),
        "active_days_30d": int(act_days),
        "n_purchase_30d": int(n_pur),
        "purchase_ratio": float(purchase_ratio),
        "days_since_last_event": int(days_evt),
        "days_since_last_purchase": int(days_pur),
        "brand_concentration_ratio": float(conc),
        "brand_switch_count_30d": int(switch),
        "total_spend_30d": float(spend),
        "activity_ratio_15d": float(ratio15),
        "price_volatility": float(vol),
        "n_events_7d": int(evt7),
        "visit_regularity": float(reg),
        "activity_trend": float(trend),
    }

    user_inputs = apply_team_rules(user_inputs)
    x_df = pd.DataFrame([[user_inputs[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)

    # 모델 예측
    with st.spinner("⚡ AI 모델 분석 중..."):
        t0 = time.time()
        if scaler is not None:
            x_scaled = scaler.transform(x_df)
            x_in = pd.DataFrame(x_scaled, columns=FEATURE_ORDER)
        else:
            arr = []
            for col in FEATURE_ORDER:
                mean_v = float(STATS[col]["mean"])
                std_v = float(STATS[col]["std"])
                raw = float(user_inputs[col])
                arr.append(0.0 if std_v == 0 else (raw - mean_v) / std_v)
            x_in = pd.DataFrame([arr], columns=FEATURE_ORDER)

        prob = predict_prob(model, x_in)
        latency_ms = (time.time() - t0) * 1000
        
        risk_level, risk_color, risk_icon, risk_bg, hit_k = risk_from_topk(prob, topk_cutoffs)
        pct_label = percentile_label(prob, pcts)
        
        # [수정] 레이더 차트 값 계산: 
        # 값이 0일 때 그래프도 0이 되도록 처리 (활동빈도, 구매액, 전환율)
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

# 3. 시각화 및 결과 화면 렌더링
categories = ["최근성", "활동빈도", "구매액", "전환율", "활동추세"]

fig_radar = go.Figure()
fig_radar.add_trace(go.Scatterpolar(
    r=radar_values,
    theta=categories,
    fill='toself',
    fillcolor=f'rgba({int(risk_color[1:3], 16)}, {int(risk_color[3:5], 16)}, {int(risk_color[5:7], 16)}, 0.25)' if is_analyzed else 'rgba(200,200,200,0.2)',
    line=dict(color=risk_color if is_analyzed else '#ccc', width=3),
    marker=dict(size=10, color=risk_color if is_analyzed else '#ccc'),
    hovertemplate='%{theta}: %{r:.1%}<extra></extra>',
    name='위험도'
))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1],
            tickformat='.0%',
            gridcolor='#e8e8e8',
            tickfont=dict(size=10)
        ),
        angularaxis=dict(gridcolor='#e8e8e8', tickfont=dict(size=11, color='#333'))
    ),
    showlegend=False,
    height=320,
    margin=dict(l=35, r=35, t=10, b=10),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
)

with col_right:
    st.markdown('<div class="section-header">🎯 예측 결과</div>', unsafe_allow_html=True)

    latency_display = f"⚡ {latency_ms:.2f} ms" if is_analyzed else "Ready"
    latency_txt_color = "#28a745" if is_analyzed else "#ccc"

    sub = []
    if hit_k is not None: sub.append(f"Top {hit_k}%")
    if pct_label: sub.append(pct_label)
    sub_txt = " | ".join(sub) if sub else "데이터를 입력하고 예측 버튼을 눌러주세요"

    st.markdown(
        f"""
        <div class="result-wrap" style="background:{risk_bg};">
            <div class="result-card" style="background: transparent; box-shadow:none; margin:0; padding: 0.2rem;">
                <div style="text-align: center;">
                    <div style="font-size: 0.8rem; color: #666; margin-bottom: 0.2rem;">{sub_txt}</div>
                    <div style="color: {risk_color if is_analyzed else '#ccc'}; font-size: 2.4rem; font-weight: 800; line-height: 1.1; margin: 0;">
                        {prob*100:.1f}%
                    </div>
                    <div style="font-size: 0.85rem; color: #888; margin-bottom: 0.5rem;">이탈 확률</div>
                    <div class="risk-badge" style="background: {risk_color}; color: {'white' if is_analyzed else '#666'}; padding: 0.3rem 1.2rem; font-size: 0.9rem; margin-top: 0;">
                        {risk_icon} {risk_level}
                    </div>
                    <div style="margin-top: 0.4rem; font-size: 0.75rem; color: {latency_txt_color}; font-weight: 600;">
                        {latency_display}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("<br>", unsafe_allow_html=True)

    t1, t2 = st.columns([2,1])
    
    with t1:
        st.markdown(
            """
            <div class="frame-head">
                <div class="frame-title">📊 위험 요인 분석</div>
            </div>
            <div class="frame-line"></div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with t2:
        st.markdown(
            """
            <div class="frame-head">
                <div class="frame-title">📈 핵심 지표</div>
            </div>
            <div class="frame-line"></div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="kpi-pane">', unsafe_allow_html=True)

        if is_analyzed:
            val_days = f"{user_inputs['days_since_last_event']}일"
            val_spend = f"{user_inputs['total_spend_30d']/10000:.0f}만원"
            val_ratio = f"{user_inputs['purchase_ratio']*100:.1f}%"
        else:
            val_days = "-"
            val_spend = "-"
            val_ratio = "-"

        # [수정] CSS 클래스명 매칭 (stat-card-small 적용)
        st.markdown(
            f"""
            <div class="kpi-wrap">
                <div class="stat-card-small">
                    <div class="stat-label">최근 활동</div>
                    <div class="stat-value">{val_days}</div>
                </div>
                <div class="stat-card-small">
                    <div class="stat-label">구매액</div>
                    <div class="stat-value">{val_spend}</div>
                </div>
                <div class="stat-card-small">
                    <div class="stat-label">전환율</div>
                    <div class="stat-value">{val_ratio}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown("</div>", unsafe_allow_html=True)

    