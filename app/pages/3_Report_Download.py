import streamlit as st
# ===== util 파일 불러오기 =======
from utils.ui import apply_base_layout, top_nav

st.set_page_config(layout="wide")

apply_base_layout()

# =============================
# 상단 네비게이션바
top_nav()
# =============================
import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go

# --------------------------------------------------------------------------------
# 1. 페이지 설정 및 스타일
# --------------------------------------------------------------------------------
st.set_page_config(page_title="Report", page_icon="🚀", layout="wide")
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
        gap: 0.9rem !important;
    }
    
    /* 4. KPI 카드 스타일 */
    .kpi-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .kpi-title { font-size: 16px; color: #6b7280; font-weight: 600; margin-bottom: 5px; }
    .kpi-value-big { font-size: 48px; color: #dc2626; font-weight: 900; line-height: 1.2; }
    .kpi-value-sub { font-size: 24px; color: #374151; font-weight: 800; }
    .kpi-note { font-size: 14px; color: #9ca3af; margin-top: 5px; }
    
    /* 5. 버튼 영역 스타일 */
    .action-area {
        background-color: #f0fdf4;
        border: 2px dashed #86efac;
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        margin-top: 10px;
    }
    
    /* 6. 테이블 폰트 조정 */
    .dataframe { font-size: 14px !important; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# 2. Mock Data (가짜 데이터 생성)
# --------------------------------------------------------------------------------
@st.cache_data
def get_target_users(k_percent, total_users=100000):
    """K%에 해당하는 타겟 유저 ID 생성"""
    count = int(total_users * (k_percent / 100))
    
    # ID만 있는 심플한 데이터프레임
    df = pd.DataFrame({
        "User_ID": [f"USER_{i:06d}" for i in range(1, count + 1)],
        "Risk_Score": np.random.uniform(0.7, 0.99, count) # 정렬용 점수
    })
    df = df.sort_values("Risk_Score", ascending=False)
    return df, count


# --------------------------------------------------------------------------------
# 3. [수정됨] 상단 컨트롤 패널 (사이드바 대신 메인 화면에 배치)
# --------------------------------------------------------------------------------
st.title("🚀 Marketing Action Dashboard")
st.markdown("예측된 이탈 위험군 규모를 확인하고, **쿠폰 발송** 또는 **리스트 다운로드**를 수행하세요.")

# 깔끔한 박스 안에 설정 기능을 넣습니다.
with st.container(border=True):
    col_set1, col_set2 = st.columns([1, 2])
    
    # [왼쪽] 모델 정보 (고정)
    with col_set1:
        st.markdown("##### ⚙️ Model Setting")
        st.info("✅ 적용 모델: **LightGBM (Best)**")
        
    # [오른쪽] 타겟 범위 선택 (셀렉트 박스로 변경!)
    with col_set2:
        st.markdown("##### 🎯 Targeting Scope")
        
        # 슬라이더 대신 셀렉트 박스 사용 (옵션 미리 정의)
        target_options = {
            5:  "상위 5% (핵심 집중 관리 - 고효율)",
            10: "상위 10% (이탈 위험군 - 권장)",
            15: "상위 15% (잠재 위험군 - 적극 방어)",
            20: "상위 20% (광범위 케어)",
            30: "상위 30% (최대 범위)"
        }
        
        # 선택된 Key값(5, 10...)을 k_percent로 받음
        k_percent = st.selectbox(
            "이탈 위험군 타겟 범위를 선택하세요:",
            options=list(target_options.keys()), # [5, 10, 15, 20, 30]
            format_func=lambda x: target_options[x], # 화면에는 설명 텍스트 표시
            index=1 # 기본값: 10%
        )

# --------------------------------------------------------------------------------
# 4. 메인 화면 (KPI Dashboard) - 로직은 그대로 연결됨
# --------------------------------------------------------------------------------

# 데이터 계산 (위에서 선택한 k_percent가 여기로 들어갑니다)
target_df, target_count = get_target_users(k_percent)
lift_value = 4.2 - (k_percent * 0.1) 

st.write("") 



# 데이터 계산
target_df, target_count = get_target_users(k_percent)
lift_value = 4.2 - (k_percent * 0.1) # 가짜 Lift (범위 넓어지면 효율 떨어짐)

st.write("") 

# === [섹션 1] 핵심 지표 (Huge KPIs) ===
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    # 가장 중요한 숫자 (크게!)
    st.markdown(f"""
    <div class="kpi-card" style="border-left: 5px solid #dc2626;">
        <div class="kpi-title">🔥 집중 관리 대상 (Potential Churners)</div>
        <div class="kpi-value-big">{target_count:,} 명</div>
        <div class="kpi-note">상위 {k_percent}% 위험군 추출 완료</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">📈 예상 마케팅 효율 (Lift)</div>
        <div class="kpi-value-sub">{lift_value:.1f} x</div>
        <div class="kpi-note">랜덤 타겟팅 대비 효율<br><br></div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    expected_save = int(target_count * 0.15) # 15% 방어 가정
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">🛡️ 이탈 방어 기대 효과</div>
        <div class="kpi-value-sub">≈ {expected_save:,} 명</div>
        <div class="kpi-note">방어율 15% 가정 시<br><br></div>
    </div>
    """, unsafe_allow_html=True)

# ================================================================================
# [섹션 2] 액션 실행 (Action Item)
# ================================================================================
st.subheader("⚡ Execute Action")

col_action, col_preview = st.columns([1.5, 2])

with col_action:
    # 액션 박스
    # st.markdown('<div class="action-area">', unsafe_allow_html=True)
    st.markdown("#### 🎁 쿠폰 일괄 발송")
    st.write(f"대상: **{target_count:,}명**")
    
    # 쿠폰 종류 선택
    coupon_type = st.selectbox(
        "발송할 쿠폰 선택",
        ["[VIP] 20% 컴백 할인 쿠폰", "[일반] 무료 배송 쿠폰", "[Warning] 5,000 포인트 지급"],
        label_visibility="collapsed"
    )
    
    st.write("")
    
    # 발송 버튼 (누르면 애니메이션)
    if st.button("🚀 쿠폰 발송하기", type="primary", use_container_width=True):
        progress_text = "대상자 추출 및 발송 서버 연결 중..."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.01) # 가짜 로딩
            my_bar.progress(percent_complete + 1, text=progress_text)
            
        time.sleep(0.5)
        my_bar.empty()
        
        st.success(f"✅ 총 {target_count:,}명에게 '{coupon_type}' 발송이 예약되었습니다!")
        st.balloons()
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.write("")
    st.markdown("#### 📂 명단 다운로드")
    # CSV 다운로드
    csv = target_df[['User_ID']].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 대상자 ID 리스트 다운로드 (.csv)",
        data=csv,
        file_name=f"Target_Users_Top{k_percent}pct.csv",
        mime="text/csv",
        use_container_width=True
    )

# ================================================================================
# [섹션 3] 명단 미리보기 (Preview)
# ================================================================================
with col_preview:
    st.markdown(f"#### 📋 타겟 리스트 미리보기 (Top {k_percent}%)")
    
    # 데이터프레임 표시 (ID만 깔끔하게)
    st.dataframe(
        target_df[['User_ID']].head(100), # 100개만 표시
        use_container_width=True,
        hide_index=True,
        height=350
    )
    st.caption(f"*보안을 위해 상위 100명의 ID만 표시됩니다. 전체 명단은 왼쪽 버튼으로 다운로드하세요.")