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

from app.utils.paths import DEFAULT_PATHS as P, ensure_runtime_dirs



# 0) Page Config + Refined CSS


st.set_page_config(
    page_title="AI 이탈 예측 솔루션",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700&display=swap');
    * { font-family: 'Noto Sans KR', sans-serif; }

    .main {
        padding: 0.5rem 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* ✅ 결과 카드(조금 더 컴팩트) */
    .result-card {
        background: white;
        padding: 1.4rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.12);
        margin: 0.6rem 0;
    }

    .result-wrap {
        border-radius: 18px;
        padding: 1.0rem;
        box-shadow: 0 10px 32px rgba(0,0,0,0.10);
        margin: 0.6rem 0;
        border: 1px solid rgba(0,0,0,0.05);
    }

    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1.6rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 0.95rem;
        margin-top: 0.7rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    .section-header {
        color: #667eea;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }

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

    /* text_input 입력창 스타일(placeholder UX 강화) */
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

    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.0rem 0.8rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(102,126,234,0.25);
    }

    .stat-value {
        font-size: 1.45rem;
        font-weight: 800;
        margin: 0.15rem 0;
    }

    .stat-label {
        font-size: 0.8rem;
        opacity: 0.95;
        font-weight: 500;
    }

    /* ✅ main 배경 안에서만 흰색 타이포 */
    .main h1 { color: white !important; font-size: 2.2rem !important; margin-bottom: 0.3rem !important; }
    .main h3 { color: rgba(255,255,255,0.9) !important; font-size: 1.1rem !important; font-weight: 400 !important; }

    /* ✅ 페이지 상단 제목(컨텐츠) */
    .page-title { color:#111 !important; font-size: 2.2rem; margin:0; }
    .page-subtitle { color:#444 !important; font-size: 1.1rem; font-weight: 400; }

    .stSelectbox label, .stTextInput label { font-weight: 600; font-size: 0.9rem; }

   <style>
/* === FRAME HEADER (dot 제거 + 균형 맞춤) === */
.frame-head{
  display:flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  margin: 0.2rem 0 0.35rem 0;
}

.frame-title{
  display:flex;
  align-items:center;
  gap:0.55rem;
  color:#667eea;
  font-size: 1.25rem;   /* ✅ 예측결과(section-header)와 맞춤 */
  font-weight: 700;
  line-height: 1.15;
}

/* 가로 라인만 남김(점 없음) */
.frame-line{
  height: 10px;
  margin: 0.15rem 0 0.8rem 0;
  position: relative;
}
.frame-line::before{
  content:"";
  position:absolute;
  left:0; right:0;
  top:50%;
  height: 4px;
  background:#667eea;
  transform: translateY(-50%);
  border-radius: 999px;
  opacity: 0.95;
}

/* === KPI 패널: 오른쪽 끝 + 세로선(border-left)로 안정화 === */
.kpi-pane{
  border-left: 4px solid #667eea;  /* ✅ 세로선 대체 */
  padding-left: 1.0rem;
  height: 100%;
}

.kpi-wrap{
  display:flex;
  flex-direction: column;
  align-items: flex-end; /* ✅ 오른쪽으로 붙이기 */
  gap: 0.6rem;
  width: 100%;
}

/* KPI 카드 더 작게 */
.stat-card.small{
  width: 150px;
  border-radius: 16px;
  padding: 0.75rem 0.6rem;
  box-shadow: 0 4px 10px rgba(102,126,234,0.20);
}

.stat-card.small .stat-label{
  font-size: 0.82rem;
  font-weight: 700;
  opacity: 0.95;
}

.stat-card.small .stat-value{
  font-size: 1.85rem;  /* ✅ 더 작게 */
  font-weight: 900;
  margin-top: 0.15rem;
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



# 3) Placeholder 기반 숫자 입력 헬퍼 (미입력=0, 입력오류만 에러)


def _parse_int_optional(label: str, raw: str, min_v: int, max_v: int):
    raw = (raw or "").strip()
    if raw == "":
        return 0, None
    try:
        v = int(float(raw))
    except Exception:
        return 0, f"{label}: 숫자만 입력해주세요."
    if v < min_v or v > max_v:
        return 0, f"{label}: 범위({min_v}~{max_v}) 안에서 입력해주세요."
    return v, None

def _parse_float_optional(label: str, raw: str, min_v: float, max_v: float):
    raw = (raw or "").strip()
    if raw == "":
        return 0.0, None
    try:
        v = float(raw)
    except Exception:
        return 0.0, f"{label}: 숫자만 입력해주세요."
    if v < min_v or v > max_v:
        return 0.0, f"{label}: 범위({min_v}~{max_v}) 안에서 입력해주세요."
    return v, None

def input_int_placeholder(label: str, key: str, min_v: int, max_v: int):
    placeholder = f"숫자를 입력해주세요 ({min_v}~{max_v})"
    raw = st.text_input(label, value="", key=key, placeholder=placeholder)
    return _parse_int_optional(label, raw, min_v, max_v)

def input_float_placeholder(label: str, key: str, min_v: float, max_v: float):
    placeholder = f"숫자를 입력해주세요 ({min_v}~{max_v})"
    raw = st.text_input(label, value="", key=key, placeholder=placeholder)
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


st.markdown(f'<h1 class="page-title">🎯 AI 고객 이탈 예측 시스템(DL | {FIXED_MODEL_NAME})</h1>', unsafe_allow_html=True)



# 7) UI - Input Form (좌:우 = 3:2)


col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown('<div class="section-header">🛍️ 구매 활동</div>', unsafe_allow_html=True)

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
                min_v=0,
                max_v=365,
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
        submit = st.form_submit_button("🚀 이탈 위험 예측하기", use_container_width=True)



# 8) Predict + Result UI


def collect_errors(*errs):
    return [e for e in errs if e is not None]

if submit:
    errors = collect_errors(
        err_n_pur,
        err_days_evt, err_n_evt, err_act_days, err_trend,
        err_spend, err_days_pur, err_purchase_ratio,
        err_conc, err_switch, err_reg, err_ratio15,
        err_evt7, err_vol
    )

    if errors:
        with col_right:
            st.markdown('<div class="section-header">⚠️ 입력 오류</div>', unsafe_allow_html=True)
            for e in errors:
                st.error(e)
        st.stop()

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

    # 레이더용 스코어
    feature_scores = {
        "최근성": min(user_inputs["days_since_last_event"] / 60, 1.0),
        "활동빈도": 1.0 - min(user_inputs["n_events_30d"] / 1000, 1.0),
        "구매액": 1.0 - min(user_inputs["total_spend_30d"] / 1_000_000, 1.0),
        "전환율": 1.0 - float(user_inputs["purchase_ratio"]),
        "활동추세": min(abs(user_inputs["activity_trend"]) / 10, 1.0) if user_inputs["activity_trend"] < 0 else 0,
    }

    categories = list(feature_scores.keys())
    values = list(feature_scores.values())

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor=f'rgba({int(risk_color[1:3], 16)}, {int(risk_color[3:5], 16)}, {int(risk_color[5:7], 16)}, 0.25)',
        line=dict(color=risk_color, width=3),
        marker=dict(size=10, color=risk_color),
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
            angularaxis=dict(
                gridcolor='#e8e8e8',
                tickfont=dict(size=11, color='#333')
            )
        ),
        showlegend=False,
         height=320,
        margin=dict(l=35, r=35, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    with col_right:
        st.markdown('<div class="section-header">🎯 예측 결과</div>', unsafe_allow_html=True)

        latency_color = "#28a745" if latency_ms < 100 else "#888"

        sub = []
        if hit_k is not None:
            sub.append(f"Top {hit_k}%")
        if pct_label:
            sub.append(pct_label)
        sub_txt = " | ".join(sub) if sub else ""

        st.markdown(
            f"""
            <div class="result-wrap" style="background:{risk_bg};">
                <div class="result-card" style="background: transparent; box-shadow:none; margin:0;">
                    <div style="text-align: center; padding: 0.6rem 0;">
                        <div style="font-size: 0.82rem; color: #666; margin-bottom: 0.5rem;">{sub_txt}</div>
                        <div style="color: {risk_color}; font-size: 3.0rem; font-weight: 800; margin: 0.35rem 0;">
                            {prob*100:.1f}%
                        </div>
                        <div style="font-size: 0.9rem; color: #888; margin-bottom: 0.55rem;">이탈 확률</div>
                        <div class="risk-badge" style="background: {risk_color}; color: white;">
                            {risk_icon} {risk_level}
                        </div>
                        <div style="margin-top: 0.7rem; font-size: 0.8rem; color: {latency_color}; font-weight: 600;">
                            ⚡ {latency_ms:.2f} ms
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ✅ 스샷 프레임: 점 3개 포함 + 헤더는 section-header와 동일 크기
        # ✅ 프레임 헤더(점 3개 제거)
        st.markdown(
            """
            <div class="frame-head">
                <div class="frame-title">📊 위험 요인 분석</div>
                <div class="frame-title">📈 핵심 지표</div>
            </div>
            <div class="frame-line"></div>
            """,
            unsafe_allow_html=True,
        )

        # ✅ 본문: (레이더) | (KPI 패널: border-left로 세로선 대체)
        radar_col, kpi_col = st.columns([2.6, 1.0], gap="medium")

        with radar_col:
            st.plotly_chart(fig_radar, use_container_width=True)

        with kpi_col:
            st.markdown('<div class="kpi-pane">', unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="kpi-wrap">
                    <div class="stat-card small">
                        <div class="stat-label">최근 활동</div>
                        <div class="stat-value">{user_inputs["days_since_last_event"]}일</div>
                    </div>

                    <div class="stat-card small">
                        <div class="stat-label">구매액</div>
                        <div class="stat-value">{user_inputs["total_spend_30d"]/10000:.0f}만원</div>
                    </div>

                    <div class="stat-card small">
                        <div class="stat-label">전환율</div>
                        <div class="stat-value">{user_inputs["purchase_ratio"]*100:.1f}%</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("</div>", unsafe_allow_html=True)