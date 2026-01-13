import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score
from utils.ui import apply_base_layout, hide_sidebar, top_nav

# [중요] set_page_config는 항상 최상단에 위치
st.set_page_config(
    page_title="Action & Report", 
    layout="wide",
    initial_sidebar_state="collapsed" 
)

apply_base_layout()
hide_sidebar()
top_nav()

# ==========================================
# [CSS 스타일링]
# ==========================================
st.markdown("""
<style>
    /* 1. 최상단 여백 제거 (네비바가 들어갈 공간 확보) */
    .block-container { 
        padding-top: 0.6rem !important;
        padding-bottom: 3rem; 
    }
    
    /* 2. [핵심] 타이틀(h1) 강제로 위로 끌어올리기 */
    h1 {
        padding-top: 1rem !important;
        margin-top: -2rem !important; /* 이 값을 조절해서 간격을 맞추세요 (-2rem ~ -4rem 추천) */
    }

    /* 3. 네비게이션 바와 본문 사이의 쓸데없는 간격 제거 */
    div[data-testid="stVerticalBlock"] {
        gap: 0.5rem !important;
    }


    /* 전체 배경 */
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* 메인 타이틀 !!!!!!!!!!!!!!!!!*/
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
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
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

    /* 쿠폰 전송 버튼 스타일 */
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
    
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(221, 36, 118, 0.5) !important;
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
</style>
""", unsafe_allow_html=True)

# ==========================================
# [경로 설정]
# ==========================================
BASE_DIR = Path("/Users/kimjiwoo/Documents/SKN23-2nd-3Team")
DATA_DIR = BASE_DIR / "data"
EVAL_SCORING = DATA_DIR / "scoring.parquet"
METRICS_PATH = DATA_DIR / "metrics.json"

def load_json(p: Path):
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}

@st.cache_data
def load_parquet(p: Path):
    return pd.read_parquet(p)

metrics = load_json(METRICS_PATH)

# Session state 초기화
if 'show_modal' not in st.session_state:
    st.session_state.show_modal = False
if 'coupon_sent' not in st.session_state:
    st.session_state.coupon_sent = False

# [핵심] 에디터 초기화용 키 생성
if 'editor_key' not in st.session_state:
    st.session_state.editor_key = 0

# ==========================================
# [헤더]
# ==========================================
# st.markdown("""
# <div class="report-header">
#     <div class="report-title">📊 Action & Report Dashboard</div>
#     <div class  ="report-subtitle">고객 이탈 예측 분석 및 타겟 쿠폰 발송 관리</div>
# </div>
# """, unsafe_allow_html=True)


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
    ⚡ Action & Report Dashboard
    </h1>
    <p style="
        font-size: 1.1rem;
        color: #6c757d;
        margin: 0;
        font-weight: 500;
        padding-bottom: 15px;
    ">
        고객 이탈 예측 분석 및 타겟 쿠폰 발송 관리
    </p>
</div>
""", unsafe_allow_html=True)
st.divider()

# ==========================================
# [설정 & KPI 영역]
# ==========================================
settings_col, kpi_col = st.columns([1, 2.5])

with settings_col:
    with st.container(border=True):
        # st.markdown('<div class="settings-title" style="margin-bottom: 0px !important;">설정</div>', unsafe_allow_html=True)
        
        b1, b2, b3 = st.columns([0.1, 2, 0.1])
        with b2:
            # 1. 스타일 정의 (위/아래 여백 동시 제어)
            st.markdown("""
            <style>
                div[data-testid="stRadio"] {
                    margin-top: -10px !important;    /* ⬆️ 핵심: 타이틀(설정)과의 간격을 줄임 */
                    margin-bottom: -5px !important; /* ⬇️ 두 라디오 버튼 사이의 간격을 줄임 */
                }
            </style>
            """, unsafe_allow_html=True)


            # 3. 위젯 배치
            mode = st.radio("모드", ["평가(test)"])
            k_percent = st.radio("Top-K (%)", [5, 10, 15, 30], horizontal=True)
            # st.markdown("<br>", unsafe_allow_html=True)

# 데이터 로드
if mode == "평가(test)":
    if not EVAL_SCORING.exists():
        st.error(f"평가용 scoring 파일이 없습니다: {EVAL_SCORING}")
        st.stop()
    df = load_parquet(EVAL_SCORING)

if "user_id" not in df.columns or "risk_score" not in df.columns:
    st.error("scoring 파일에는 최소 user_id, risk_score 컬럼이 필요합니다.")
    st.stop()

df["user_id"] = df["user_id"].astype(str)
df_sorted = df.sort_values("risk_score", ascending=False).reset_index(drop=True)

n = len(df_sorted)
k = max(int(np.ceil(n * (k_percent / 100))), 1)
topk = df_sorted.head(k).copy()



with kpi_col:
    with st.container(border=True):
        st.markdown("##### **성능 지표**")
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
        c1.metric("PR-AUC", f"{ap:.4f}", f"+{(ap-0.5):.3f}")
        c2.metric(f"Precision@{k_percent}%", f"{precision_k:.4f}", f"{(precision_k/base_rate):.1f}x")
        c3.metric(f"Recall@{k_percent}%", f"{recall_k:.4f}", f"{captured_pos:,}명")
        c4.metric(f"Lift@{k_percent}%", f"{lift_k:.2f}", "개선도")


with st.expander("📊 세부 지표", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"- **전체 대상**: {n:,}명\n- **Top {k_percent}%**: {k:,}명")
    with col2:
        st.markdown(f"- **포착 이탈**: {captured_pos:,}명\n- **포착률**: {(captured_pos/total_pos*100):.1f}%")

st.divider()


# ==========================================
# [메인 컨텐츠 - 리스트 & 액션]
# ==========================================
left_col, right_col = st.columns([1.8, 1])

# ----------------------------------------------------------------
# [왼쪽] 데이터 에디터 (체크박스 기능 추가)
# ----------------------------------------------------------------
with left_col:
    st.markdown(f'<div class="section-title">Top {k_percent}% 발송 대상자</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="margin-bottom: 1rem;">
        <span class="stat-badge">전체 {n:,}명</span>
        <span class="stat-badge">발송 {k:,}명</span>
        <span class="stat-badge">예상 이탈률 {(topk["y_true"].mean()*100):.1f}%</span>
    </div>
    """, unsafe_allow_html=True)
    
    show_cols = ["user_id", "risk_score"]
    # if "y_true" in topk.columns:
    #     show_cols.append("y_true")
    
    # 1. 데이터 준비
    display_df = topk[show_cols].head(20).copy()
    
    # [수정] '선택' 컬럼을 데이터프레임 맨 뒤(오른쪽)에 추가
    display_df["선택"] = False
    
    # 2. Data Editor 생성
    edited_df = st.data_editor(
        display_df,
        column_config={
            "선택": st.column_config.CheckboxColumn(
                "선택", 
                default=False, 
                width=5 # [수정] 너비를 50px로 고정하여 딱 맞게 설정
            ),
            "risk_score": st.column_config.NumberColumn("위험 점수", format="%.4f")
        },
        disabled=show_cols, # 기존 컬럼은 수정 불가
        use_container_width=True,
        hide_index=True,
        height=450,
        key=f"data_editor_{st.session_state.editor_key}" # 초기화용 키
    )

    a1, a3 = st.columns([3, 1])
    with a1:
        st.caption(f"💡 체크박스를 선택하여 개별 발송하거나, 미선택 시 Top {k_percent}% 전원에게 발송합니다.")
    with a3:
        csv = topk[["user_id", "risk_score"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Top {k_percent}% Download (.csv)",
            data=csv,
            file_name=f"top_{k_percent}pct_users.csv",
            mime="text/csv",
            use_container_width=True
        )
# ----------------------------------------------------------------
# [오른쪽] 액션 센터
# ----------------------------------------------------------------
with right_col:
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="section-title">쿠폰 발송</div>', unsafe_allow_html=True)
        
        # 1. 전체 발송 가능한 쿠폰 리스트 정의
        all_coupons = [
            "30% 특별 할인 쿠폰",
            "20% 프리미엄 할인 쿠폰",
            "15% 할인 쿠폰",
            "10% 할인 쿠폰",
            "5,000원 장바구니 쿠폰",
            "배송비 무료 쿠폰",
            "[시크릿] 24시간 타임 쿠폰"
        ]
        
        # 2. Top-K 비율에 따른 추천 쿠폰 매핑 (자동 추천 로직)
        # (여기서 정의한 쿠폰 이름이 위 all_coupons 리스트에 있어야 합니다)
        recommend_map = {
            5: "30% 특별 할인 쿠폰",
            10: "20% 프리미엄 할인 쿠폰",
            15: "15% 할인 쿠폰",
            30: "10% 할인 쿠폰"
        }
        
        # 3. 기본 선택값(Default Index) 설정
        # 현재 k_percent에 맞는 쿠폰을 찾아서 셀렉트박스의 기본값으로 설정
        recommended_coupon = recommend_map.get(k_percent, "10% 할인 쿠폰")
        try:
            default_index = all_coupons.index(recommended_coupon)
        except ValueError:
            default_index = 3 # 리스트에 없으면 안전하게 10% 쿠폰 선택



        # 4. 쿠폰 선택 셀렉트 박스 [추가된 부분]
        selected_coupon_final = st.selectbox(
            "발송할 쿠폰 선택 (기본값: 추천 쿠폰)",
            options=all_coupons,
            index=default_index
        )
        
        # -----------------------------------------------------------
        
        # 타겟 인원 계산 (왼쪽 에디터 연동)
        selected_rows = edited_df[edited_df["선택"] == True]
        is_selection_mode = not selected_rows.empty
        
        target_count = len(selected_rows) if is_selection_mode else k
        target_text = f"✅ 선택된 {target_count}명" if is_selection_mode else f"Top {k_percent}% ({target_count:,}명)"

        # 전략 설명 (오디언스 기준)
        audience_strategy = {
            5: "🌟 최고 위험군 케어",
            10: "⭐ 고위험군 이탈 방지",
            15: "💫 중위험군 혜택 제공",
            30: "✨ 잠재 위험군 관리"
        }

        # 정보 박스 표시 (선택된 쿠폰 반영)
        st.markdown(f"""
        <div class="coupon-info-box">
            <div class="coupon-info-title">📬 발송 정보 확인</div>
            <div class="coupon-info-item">• 대상: <strong>{target_text}</strong></div>
            <div class="coupon-info-item">• 쿠폰: <strong>{selected_coupon_final}</strong></div>
            <div class="coupon-info-item">• 타겟 전략: {audience_strategy.get(k_percent, '일반 관리')}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()

        # 전송 버튼
        if st.button("🚀 쿠폰 전송하기", type="primary", use_container_width=True):
            st.session_state.show_modal = True
            
            # [핵심] 셀렉트 박스에서 최종 선택된 쿠폰을 저장
            st.session_state.sent_coupon_type = selected_coupon_final 
            st.session_state.sent_k_percent = k_percent
            
            # 선택 모드에 따라 전송 인원 저장
            st.session_state.sent_k = target_count
            
            # 전송 후 에디터 초기화를 위해 키 값 증가
            st.session_state.editor_key += 1
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
# ==========================================
# [모달]
# ==========================================
if st.session_state.show_modal:
    # 수정된 부분: coupon_type 변수가 없으므로 기본값을 문자열로 대체하거나 session_state만 참조
    sent_coupon = st.session_state.get('sent_coupon_type', "쿠폰") 
    sent_k_percent = st.session_state.get('sent_k_percent', k_percent)
    sent_k = st.session_state.get('sent_k', k)
    
    st.markdown(f"""
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
    """, unsafe_allow_html=True)
    
    time.sleep(2)
    st.session_state.show_modal = False
    st.rerun()