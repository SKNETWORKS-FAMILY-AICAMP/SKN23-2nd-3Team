import json
from pathlib import Path
import time
import io  # [필수] 텍스트 파일 생성을 위해 필요

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score
from utils.ui import apply_base_layout, hide_sidebar, top_nav

# [중요] set_page_config는 항상 최상단에 위치
st.set_page_config(
    page_title="Action & Report", layout="wide", initial_sidebar_state="collapsed"
)

apply_base_layout()
hide_sidebar()
top_nav()

# ============ 간격 조정 =============
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# CSS 설정
st.markdown(
    """
<style>
   /* 전체 배경 */
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* 타이틀 영역 */
    .report-header {
        background: white;
        padding: 1rem;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
        border-left: 4px solid #3b82f6;
    }
    
    .report-title  {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #dd2e1f 20%, #ffdff6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 30px;
        text-align: left;
    }

    /* 설정 카드 */
    .settings-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1rem;
    }
    
    /* 메트릭 스타일 */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
    }
    
    div[data-testid="stMetricDelta"] {
        font-size: 0.85rem;
    }
    
    /* 섹션 타이틀 */
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* 쿠폰 정보 박스 */
    .coupon-info-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        padding: 1.25rem;
        border-radius: 12px;
        border: 2px solid #bfdbfe;
        margin: 1rem 0;
    }
    
    .coupon-info-title {
        font-weight: 700;
        color: #1e40af;
        margin-bottom: 0.75rem;
        font-size: 1rem;
    }
    
    .coupon-info-item {
        color: #1e40af;
        margin: 0.5rem 0;
        font-size: 0.95rem;
    }
    /* 쿠폰 전송 버튼 스타일 (기본) */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #FF512F 0%, #DD2476 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        box-shadow: 0 4px 15px rgba(221, 36, 118, 0.3) !important;
        transition: all 0.3s ease !important;
    }

    /* 쿠폰 전송 버튼 스타일 (Hover 시 밑줄 제거 및 효과) */
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(221, 36, 118, 0.5) !important;
        color: white !important; /* 글자색 흰색 유지 */
        text-decoration: none !important; /* 밑줄 제거 핵심 코드 */
    }

    .stButton > button:hover::after{
        display: none;
    }

    /* 모달 */
    .modal-overlay {
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0, 0, 0, 0.6);
        display: flex; justify-content: center; align-items: center;
        z-index: 9999; backdrop-filter: blur(4px);
    }
    .modal-content {
        background: white; padding: 3rem 2.5rem; border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3); text-align: center;
        max-width: 450px; animation: modalSlide 0.3s ease;
    }
    @keyframes modalSlide {
        from { opacity: 0; transform: translateY(-30px) scale(0.95); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    .modal-icon { font-size: 4rem; margin-bottom: 1rem; }
    .modal-title { font-size: 1.6rem; font-weight: 700; color: #1e293b; margin-bottom: 1rem; }
    .modal-message { font-size: 1rem; color: #64748b; line-height: 1.6; }
    
    /* 통계 배지 */
    .stat-badge {
        display: inline-block; background: #f1f5f9; padding: 0.5rem 1rem;
        border-radius: 8px; font-weight: 600; color: #475569; margin: 0.25rem;
    }

    /* 다운로드 버튼 스타일 커스텀 */
    div.stDownloadButton > button {
        font-size: 0.8rem !important; 
        border: none !important;
        box-shadow: none !important; 
        background-color: #fafafa !important;
        padding: 5px 15px !important;
        min-height: 0px !important;
        height: auto !important;
        line-height: 1.2 !important;
        color: #555 !important;
    }
    div.stDownloadButton > button:hover {
        background-color: #e0e2e6 !important;
        color: #333 !important;
    }
    
    /* 카드 내 버튼 스타일 (Nav Bar 버튼과 구분) */
    div[data-testid="stVerticalBlockBorderWrapper"] .stButton > button {
        background: white !important;
        border: 1px solid #e5e7eb !important;
        color: #374151 !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
        transition: all 0.2s ease !important;
        padding: 0.5rem 1rem !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] .stButton > button:hover {
        border-color: #3b82f6 !important;
        color: #2563eb !important;
        background: #eff6ff !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.2) !important;
    }
    .st-emotion-cache-pk3c77 p{ margin-bottom: 0; }

    /* Expander (아코디언) 전체 박스 스타일 */
    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        overflow: hidden; /* 모서리 둥글게 유지 */
        margin-bottom: 1rem;
    }

    /* Expander 헤더 (접혔을 때 보이는 부분) 배경색 변경 */
    div[data-testid="stExpander"] summary {
        background-color: #f8fafc !important; /* 연한 회색 배경 */
        color: #1e293b !important;            /* 글자색 진하게 */
        font-weight: 600 !important;
        border-radius: 12px;                  /* 닫혀있을 때 둥글게 */
        transition: background-color 0.2s;    /* 호버 효과 부드럽게 */
    }

    /* Expander 헤더에 마우스 올렸을 때 (Hover) */
    div[data-testid="stExpander"] summary:hover {
        background-color: #f1f5f9 !important; /* 호버 시 조금 더 진한 회색 */
    }

    /* Expander가 열렸을 때 내용물 배경은 흰색 유지 */
    div[data-testid="stExpander"][aria-expanded="true"] summary {
        border-bottom-left-radius: 0 !important;
        border-bottom-right-radius: 0 !important;
        border-bottom: 1px solid #e2e8f0; /* 열리면 헤더 아래에 선 추가 */
    }
    
    div[data-testid="stExpanderDetails"] {
        background-color: white !important;
    }
 

</style>
""",
    unsafe_allow_html=True,
)

# ==========================================
# [경로 설정]
# ==========================================
BASE_DIR = Path("/Users/kimjiwoo/Documents/SKN23-2nd-3Team")
DATA_DIR = BASE_DIR / "data/processed"
EVAL_SCORING = DATA_DIR / "scoring.parquet"
FEATURES_PATH = DATA_DIR / "features_ml_clean.parquet"
METRICS_PATH = DATA_DIR / "metrics.json"


def load_json(p: Path):
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


@st.cache_data
def load_parquet(p: Path):
    return pd.read_parquet(p)


metrics = load_json(METRICS_PATH)

COLUMN_MAP = {
    "user_id": "유저 ID",
    "anchor_time": "기준 일자",
    "risk_score": "이탈 위험 점수",
    "y_true": "실제 이탈 여부",
    "n_events_30d": "최근 1달 이벤트 수",
    "active_days_30d": "최근 1달 활동 일수",
    "n_purchase_30d": "최근 1달 구매 횟수",
    "purchase_ratio": "구매 전환율",
    "total_spend_30d": "최근 1달 총 결제액",
    "n_events_7d": "최근 1주 이벤트 수",
    "days_since_last_event": "마지막 활동 경과일",
    "days_since_last_purchase": "마지막 구매 경과일",
    "visit_regularity": "방문 규칙성",
    "activity_trend": "활동 추세",
    "brand_concentration_ratio": "브랜드 집중도",
    "brand_switch_count_30d": "브랜드 교차 수",
    "price_volatility": "가격 변동성",
    "activity_ratio_15d": "최근 15일 활동 비중",
}


def build_analysis_text(user_history: pd.DataFrame) -> str:
    if user_history.empty:
        return "히스토리 없음 (심층 분석 필요)"

    last = user_history.iloc[-1]
    reasons = []

    if "n_purchase_30d" in last and float(last["n_purchase_30d"]) == 0:
        reasons.append("최근 30일간 구매 이력 없음")
    if "active_days_30d" in last and float(last["active_days_30d"]) <= 3:
        reasons.append("최근 30일간 활동 일수 매우 저조(≤3)")
    if "days_since_last_event" in last and float(last["days_since_last_event"]) >= 30:
        reasons.append("마지막 활동이 오래됨(≥30일)")
    if "visit_regularity" in last and float(last["visit_regularity"]) < 0.3:
        reasons.append("방문 규칙성이 낮음(<0.3)")
    if "activity_trend" in last and float(last["activity_trend"]) < 0:
        reasons.append("활동 추세가 하락(negative trend)")

    # [수정 핵심] "  \n" (공백 2개 + 개행)을 사용해야 Markdown에서 줄바꿈이 됩니다.
    return "- " + "\n- ".join(reasons) if reasons else "특이 패턴 미발견 (심층 분석 필요)"


# [추가] 각 구간별 상세 리포트 텍스트 생성 함수
def get_detailed_report(group_name: str) -> str:
    if "Top 5%" in group_name:
        return """
[Top 5% 구간 분석]
1. 현황 진단
   - 해당 고객은 현재 이탈 위험도가 최고 수준(Risk Score Top 5%)에 도달했습니다.
   - 모든 활동 지표가 멈춘 상태로, 사실상 서비스 이용을 중단한 '이탈 확정' 단계로 진단됩니다.

2. 상세 원인 분석
   - 구매 활동: 최근 30일간 38건의 구매가 발생했으나, 추세 확인이 필요합니다.
   - 저조한 활동: 최근 30일간 접속일이 1일에 불과하여, 매우 수동적인 이용 형태를 보입니다.
   - 불규칙한 패턴: 방문 주기가 매우 불규칙하여, 예측 불가능한 이탈 가능성이 높습니다.

3. 추천 해결방안
   - 일반적인 넛지(Nudge)로는 반응하지 않을 가능성이 높습니다.
   - '파격적인 Win-back 오퍼(예: 50% 할인, 무조건 무료 배송)'와 같은 강력한 동기 부여가 필수적입니다.
"""
    elif "Top 10%" in group_name:
        return """
[Top 10% 구간 분석]
1. 현황 진단
   - 해당 고객은 최근 서비스 방문 빈도가 급격히 감소하고 있습니다.
   - 아직 간헐적인 접속은 확인되나, 구매나 탐색과 같은 '유의미한 활동'이 실종된 상태입니다.

2. 상세 원인 분석
   - 구매 정체: 최근 30일간 구매 내역이 전무(0건)합니다. 마지막 구매로부터 31일이 경과했습니다.
   - 저조한 활동: 최근 30일간 접속일이 1일에 불과하여, 매우 수동적인 이용 형태를 보입니다.
   - 불규칙한 패턴: 방문 주기가 매우 불규칙하여, 예측 불가능한 이탈 가능성이 높습니다.

3. 추천 해결방안
   - 고객이 관심을 가질만한 개인화된 상품 추천 푸시가 필요합니다.
   - '최근 본 상품 가격 인하' 알림 등을 통해 즉각적인 재방문을 유도해야 합니다.
"""
    elif "Top 15%" in group_name:
        return """
[Top 15% 구간 분석]
1. 현황 진단
   - 이전 대비 체류 시간이 감소하고, 브랜드에 대한 집중도가 흩어지고 있습니다.
   - 타 플랫폼이나 경쟁사로의 비교 탐색(Churning)을 시작했을 가능성이 있습니다.

2. 상세 원인 분석
   - 구매 정체: 최근 30일간 구매 내역이 전무(0건)합니다. 마지막 구매로부터 31일이 경과했습니다.
   - 저조한 활동: 최근 30일간 접속일이 1일에 불과하여, 매우 수동적인 이용 형태를 보입니다.
   - 불규칙한 패턴: 방문 주기가 매우 불규칙하여, 예측 불가능한 이탈 가능성이 높습니다.

3. 추천 해결방안
   - 브랜드 로열티를 상기시킬 수 있는 접근이 필요합니다.
   - 장바구니에 담아둔 상품 리마인드나 '회원 전용 혜택' 강조가 효과적일 수 있습니다.
"""
    elif "Top 30%" in group_name:
        return """
[Top 30% 구간 분석]
1. 현황 진단
   - 전반적으로 양호한 활동을 보이고 있으나, 최근 방문 주기가 불규칙해지는 패턴이 감지되었습니다.
   - 안정적인 리텐션 유지를 위한 선제적인 관리가 요구됩니다.

2. 상세 원인 분석
   - 구매 정체: 최근 30일간 구매 내역이 전무(0건)합니다. 마지막 구매로부터 31일이 경과했습니다.
   - 저조한 활동: 최근 30일간 접속일이 1일에 불과하여, 매우 수동적인 이용 형태를 보입니다.
   - 불규칙한 패턴: 방문 주기가 매우 불규칙하여, 예측 불가능한 이탈 가능성이 높습니다.

3. 추천 해결방안
   - 정기적인 뉴스레터나 가벼운 출석 체크 이벤트를 통해 꾸준한 접점을 유지하는 것이 좋습니다.
"""
    return ""


def render_history_table(user_history: pd.DataFrame):
    if user_history.empty:
        st.info("표시할 히스토리가 없습니다.")
        return

    view_df = user_history.tail(50).copy()
    view_df = view_df.rename(columns=COLUMN_MAP)

    n_rows = len(view_df)

    if n_rows <= 2:
        last = (
            view_df.iloc[-1]
            .to_frame("값")
            .reset_index()
            .rename(columns={"index": "항목"})
        )
        st.dataframe(
            last,
            use_container_width=True,
            hide_index=True,
            height=520,
        )
        return

    auto_h = min(34 * n_rows + 40, 620)

    st.dataframe(
        view_df,
        use_container_width=True,
        hide_index=True,
        height=auto_h,
    )


# --------------------------------------------------------------------------------
# ✅ 모달(st.dialog)
# --------------------------------------------------------------------------------
def _set_modal_payload(
    group_name: str, uid: str, user_score: float, user_history: pd.DataFrame
):
    st.session_state["_modal_group_name"] = group_name
    st.session_state["_modal_uid"] = uid
    st.session_state["_modal_score"] = (
        float(user_score) if np.isfinite(user_score) else float("nan")
    )
    st.session_state["_modal_history"] = user_history


@st.dialog("상세 분석 결과", width="large")
def open_user_modal():
    group_name = st.session_state.get("_modal_group_name", "")
    uid = st.session_state.get("_modal_uid", "")
    user_score = st.session_state.get("_modal_score", float("nan"))
    user_history = st.session_state.get("_modal_history", pd.DataFrame())

    # 기본 분석 텍스트 (화면 표시용)
    analysis_text = build_analysis_text(user_history)
    
    # [추가] TXT 파일에만 들어갈 상세 리포트 텍스트 생성
    detailed_report_text = get_detailed_report(group_name)

    left, right = st.columns([1.5, 3], gap="large")

    with left:
        st.markdown(f"### 🔍 {group_name}")
        st.caption(f"User ID: {uid}")

        m1, m2 = st.columns(2)
        with m1:
            st.metric(
                "Risk Score", f"{user_score:.4f}" if np.isfinite(user_score) else "-"
            )
        with m2:
            st.metric("로그 수", f"{len(user_history):,}")

        # [요청 반영] 화면에는 기본 AI Insight만 표시
        st.info(f"💡 **AI 분석 Insight:**\n\n{analysis_text}")

        # [요청 반영] TXT 파일에는 상세 리포트 포함
        if not user_history.empty:
            txt_buffer = io.StringIO()
            txt_buffer.write(f"=== 사용자 이탈 예측 심층 리포트 ===\n\n")
            txt_buffer.write(f"User ID: {uid}\n")
            txt_buffer.write(f"Risk Group: {group_name}\n")
            txt_buffer.write(f"Risk Score: {user_score:.4f}\n")
            txt_buffer.write("-" * 50 + "\n")
            
            # 1. 기본 분석 (화면에 보이는 내용)
            txt_buffer.write(f"\n[💡 AI 기본 분석]\n")
            txt_buffer.write(f"{analysis_text}\n")
            
            # 2. 상세 리포트 (TXT 전용 내용)
            if detailed_report_text:
                txt_buffer.write(f"\n{detailed_report_text}\n")
            
            # 3. 데이터 로그
            txt_buffer.write("-" * 50 + "\n")
            txt_buffer.write(f"\n[📊 활동 로그 데이터 (CSV 포맷)]\n")
            user_history.to_csv(txt_buffer, index=False)
            
            txt_data = txt_buffer.getvalue().encode("utf-8")

            st.download_button(
                "⬇ Report(.txt) Download",
                data=txt_data,
                file_name=f"{uid}_detailed_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

    with right:
        st.markdown("#### 📊 활동 로그")
        render_history_table(user_history)


# Session state 초기화
if "show_modal" not in st.session_state:
    st.session_state.show_modal = False
if "coupon_sent" not in st.session_state:
    st.session_state.coupon_sent = False
if "editor_key" not in st.session_state:
    st.session_state.editor_key = 0

# ==========================================
# [헤더]
# ==========================================
# st.markdown(
#     """
# <div style="padding-bottom: 0px;">
#     <h1 style="
#         font-family: 'Helvetica Neue', sans-serif;
#         font-weight: 900;
#         font-size: 3rem;
#         background: linear-gradient(135deg, #dd2e1f 20%, #ffdff6 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         margin: 0;
#         padding-bottom: 5px;
#         padding-top: 10px;
#     ">
#     인사이트 & 액션 대시보드
#     </h1>
#     <p style="
#         font-size: 1.1rem;
#         color: #6c757d;
#         margin: 0;
#         font-weight: 500;
#         padding-bottom: 15px;
#     ">
#         고객 이탈 예측 분석 및 타겟 쿠폰 발송 관리
#     </p>
# </div>
# """,
#     unsafe_allow_html=True,
# )

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
        <h1 class="main-title">인사이트 & 액션 대시보드</h1>
        <p class="subtitle">고객 이탈 예측 분석 및 타겟 쿠폰 발송 관리</p>
        <div class="accent-line"></div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)
# st.divider()

# ==========================================
# [설정 & KPI 영역]
# ==========================================
settings_col, kpi_col = st.columns([1, 3])


# [2] 설정(Settings) 컨테이너 적용
with settings_col:
    # with st.container(border=True):
    #     st.markdown('<div class="settings-container"></div>', unsafe_allow_html=True)

    #     b1, b2, b3 = st.columns([0.1, 2, 0.1])
    #     with b2:
    #         # [수정] 주석 해제하여 NameError 해결
    #         # mode = st.radio("모드", ["평가(test)"])
    #         st.markdown("  ")
    #         k_percent = st.radio("Top-K (%)", [5, 10, 15, 30], horizontal=True)

    b1, b2 = st.columns([0.25, 2])
    with b2:
        st.markdown("<br>", unsafe_allow_html=True)
        k_percent = st.radio("Top-K (%)", [5, 10, 15, 30], horizontal=True)        
            

# 데이터 로드
# if mode == "평가(test)":
#     if not EVAL_SCORING.exists():
#         st.error(f"평가용 scoring 파일이 없습니다: {EVAL_SCORING}")
#         st.stop()
    df = load_parquet(EVAL_SCORING)

    if FEATURES_PATH.exists():
        df_features = load_parquet(FEATURES_PATH)
        df_features["user_id"] = df_features["user_id"].astype(str)
    else:
        df_features = pd.DataFrame(columns=["user_id"])

if "user_id" not in df.columns or "risk_score" not in df.columns:
    st.error("scoring 파일에는 최소 user_id, risk_score 컬럼이 필요합니다.")
    st.stop()

df["user_id"] = df["user_id"].astype(str)
df_sorted = df.sort_values("risk_score", ascending=False).reset_index(drop=True)

# [대표 유저 선정 로직]
df_churn = df_sorted[df_sorted["y_true"] == 1].copy().reset_index(drop=True)
total_churn = len(df_churn)
representative_ids = {}
if total_churn > 0:
    def safe_idx(x: int) -> int:
        return max(0, min(x, total_churn - 1))

    representative_ids = {
        "Top 5%": df_churn.iloc[safe_idx(int(total_churn * 0.05))]["user_id"],
        "Top 10%": df_churn.iloc[safe_idx(int(total_churn * 0.10))]["user_id"],
        "Top 15%": df_churn.iloc[safe_idx(int(total_churn * 0.15))]["user_id"],
        "Top 30%": df_churn.iloc[safe_idx(int(total_churn * 0.30))]["user_id"],
    }

n = len(df_sorted)
k = max(int(np.ceil(n * (k_percent / 100))), 1)
topk = df_sorted.head(k).copy()


with kpi_col:
    # [2] 컨테이너 코드 수정
    with st.container(border=True):
        # 👇 [핵심] 이 줄을 추가하여 위 CSS가 이 컨테이너를 찾을 수 있게 합니다.
        st.markdown('<div class="kpi-metric-container"></div>', unsafe_allow_html=True)

        _, v1, v2 = st.columns([0.05, 0.2, 1])
        
        with v1:
            # 중앙 정렬을 위해 불필요한 br은 제거하거나 조정 가능
            st.markdown("##### **성능 지표**")

        with v2:
            if "y_true" not in df_sorted.columns:
                st.error("평가(test) 모드에는 y_true가 필요합니다.")
                st.stop()

            y_true = df_sorted["y_true"].astype(int).values
            score = df_sorted["risk_score"].astype(float).values
            ap = float(average_precision_score(y_true, score))

            precision_k = float(topk["y_true"].mean())
            total_pos = int(y_true.sum())
            captured_pos = int(topk["y_true"].sum())
            recall_k = float(captured_pos / total_pos) if total_pos > 0 else 0.0
            base_rate = float(y_true.mean())
            lift_k = float(precision_k / base_rate) if base_rate > 0 else float("nan")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("PR-AUC", f"{ap:.3f}", f"+{(ap-0.5):.3f}")
            c2.metric(f"Precision@{k_percent}%", f"{precision_k:.3f}", f"{(precision_k/base_rate):.1f}x")
            c3.metric(f"Recall@{k_percent}%", f"{recall_k:.3f}", f"{captured_pos:,}명")
            c4.metric(f"Lift@{k_percent}%", f"{lift_k:.2f}", "개선도")

st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📈 세부 지표", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"- **전체 대상**: {n:,}명\n- **Top {k_percent}%**: {k:,}명")
    with col2:
        st.markdown(
            f"- **포착 이탈**: {captured_pos:,}명\n- **포착률**: {(captured_pos/total_pos*100):.1f}%"
        )
    st.divider()

# ==========================================
# [NEW SECTION] 대표 이탈 케이스 (아코디언)
# ==========================================
with st.expander("대표 이탈 케이스"):
    if not representative_ids:
        st.info("대표 유저를 선정할 수 없습니다.")
    else:
        # 가로로 4개 배치
        cols = st.columns(4)
        for i, (group_name, uid) in enumerate(representative_ids.items()):
            with cols[i]:
                score_row = df_sorted.loc[df_sorted["user_id"] == uid]
                user_score = (
                    float(score_row["risk_score"].iloc[0])
                    if not score_row.empty
                    else float("nan")
                )

                user_history = df_features.loc[df_features["user_id"] == uid].copy()
                if not user_history.empty and "anchor_time" in user_history.columns:
                    user_history = user_history.sort_values("anchor_time")

                with st.container(border=True):
                    st.subheader(group_name)
                    st.caption(f"User ID: {uid}")
                    st.metric("Risk Score", f"{user_score:.4f}")
                    st.write("") # 간격
                    
                    if st.button(
                        "Review",
                        key=f"btn_modal_{i}_{uid}",
                        use_container_width=True,
                    ):
                        _set_modal_payload(group_name, uid, user_score, user_history)
                        open_user_modal()

st.divider()

# ==========================================
# [메인 컨텐츠 - 리스트 & 액션]
# ==========================================
# 기존 3단 레이아웃 [2, 1, 1] -> [2.2, 1] 로 변경
col_editor, col_actions = st.columns([2.2, 1], gap="medium")

# ----------------------------------------------------------------
# [왼쪽] 데이터 에디터 (Top K% 발송 대상자)
# ----------------------------------------------------------------
with col_editor:
    st.markdown(
        f'<div class="section-title">Top {k_percent}% 발송 대상자</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
    <div style="margin-bottom: 1rem;">
        <span class="stat-badge">전체 {n:,}명</span>
        <span class="stat-badge">발송 {k:,}명</span>
        <span class="stat-badge" style="color: #dc3545;">예상 이탈률 {(topk["y_true"].mean()*100):.1f}%</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    show_cols = ["user_id", "risk_score"]

    display_df = topk[show_cols].head(20).copy()
    display_df["risk_score"] = display_df["risk_score"] * 100
    display_df["선택"] = False

    edited_df = st.data_editor(
        display_df,
        column_config={
            "선택": st.column_config.CheckboxColumn(
                "선택",
                default=False,
                width="small",
            ),
            "risk_score": st.column_config.NumberColumn(
                "위험 지수", 
                format="%.1f%%"
            ),
        },
        disabled=show_cols,
        use_container_width=True,
        hide_index=True,
        height=450,
        key=f"data_editor_{st.session_state.editor_key}",
    )

    a1, a3 = st.columns([3, 1])
    with a1:
        st.caption(
            f"💡 체크박스를 선택하여 개별 발송하거나, 미선택 시 Top {k_percent}% 전원에게 발송합니다."
        )
    with a3:
        csv = topk[["user_id", "risk_score"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Top {k_percent}% Download (.csv)",
            data=csv,
            file_name=f"top_{k_percent}pct_users.csv",
            mime="text/csv",
            use_container_width=True,
        )



# ----------------------------------------------------------------
# [오른쪽] 액션 센터 (쿠폰 발송)
# ----------------------------------------------------------------
with col_actions:
    # 위쪽 여백 조정 (제목 높이 맞추기 위해)
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown(
            '<div class="section-title">쿠폰 발송</div>', unsafe_allow_html=True
        )

        all_coupons = [
            "30% 특별 할인 쿠폰",
            "20% 프리미엄 할인 쿠폰",
            "15% 할인 쿠폰",
            "10% 할인 쿠폰",
            "5,000원 장바구니 쿠폰",
            "배송비 무료 쿠폰",
            "[시크릿] 24시간 타임 쿠폰",
        ]

        recommend_map = {
            5: "30% 특별 할인 쿠폰",
            10: "20% 프리미엄 할인 쿠폰",
            15: "15% 할인 쿠폰",
            30: "10% 할인 쿠폰",
        }

        recommended_coupon = recommend_map.get(k_percent, "10% 할인 쿠폰")
        try:
            default_index = all_coupons.index(recommended_coupon)
        except ValueError:
            default_index = 3

        selected_coupon_final = st.selectbox(
            "발송할 쿠폰 선택 (기본값: 추천 쿠폰)",
            options=all_coupons,
            index=default_index,
        )

        selected_rows = edited_df[edited_df["선택"] == True]
        is_selection_mode = not selected_rows.empty

        target_count = len(selected_rows) if is_selection_mode else k
        target_text = (
            f"✅ 선택된 {target_count}명"
            if is_selection_mode
            else f"Top {k_percent}% ({target_count:,}명)"
        )

        audience_strategy = {
            5: "🌟 최고 위험군 케어",
            10: "⭐ 고위험군 이탈 방지",
            15: "💫 중위험군 혜택 제공",
            30: "✨ 잠재 위험군 관리",
        }

        st.markdown(
            f"""
        <div class="coupon-info-box">
            <div class="coupon-info-title">📬 발송 정보 확인</div>
            <div class="coupon-info-item">• 대상: <strong>{target_text}</strong></div>
            <div class="coupon-info-item">• 쿠폰: <strong>{selected_coupon_final}</strong></div>
            <div class="coupon-info-item">• 타겟 전략: {audience_strategy.get(k_percent, '일반 관리')}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.divider()

        if st.button("쿠폰 전송하기", type="primary", use_container_width="stretch"):
            st.session_state.show_modal = True
            st.session_state.sent_coupon_type = selected_coupon_final
            st.session_state.sent_k_percent = k_percent
            st.session_state.sent_k = target_count
            st.session_state.editor_key += 1
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# [모달]
# ==========================================
if st.session_state.show_modal:
    sent_coupon = st.session_state.get("sent_coupon_type", "쿠폰")
    sent_k_percent = st.session_state.get("sent_k_percent", k_percent)
    sent_k = st.session_state.get("sent_k", k)

    st.markdown(
        f"""
    <div class="modal-overlay">
        <div class="modal-content">
            <div class="modal-icon">✅</div>
            <div class="modal-title">발송 완료!</div>
            <div class="modal-message">
                고객 <strong>{sent_k:,}명</strong>에게<br>
                <strong>{sent_coupon}</strong>을(를)<br>
                성공적으로 발송했습니다.
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    time.sleep(2)
    st.session_state.show_modal = False
    st.rerun()