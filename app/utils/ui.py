# utils/ui.py
import streamlit as st

# ======= 페이지 여백 확장 ========
def apply_base_layout():
    st.markdown("""
    <style>
    /* 전체 컨테이너 */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100%;
    }

    /* 헤더 제거 */
    header[data-testid="stHeader"] {
        display: none;
    }

    /* 사이드바 제거 */
    [data-testid="stSidebar"] {
        display: none;
    }

    /* 푸터 제거 */
    footer {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)


# ======== 사이드바 숨김 ===========
def hide_sidebar():
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        display: none;
        opacity: 0;
    }
    </style>
    """, unsafe_allow_html=True)


# ==== 상단 네비게이션 바 ===========
def top_nav():
    st.markdown("""
    <style>
    /* 기본 Streamlit 패딩 제거 */
    .block-container {
        padding-top: 0rem;
    }
    
    /* 네비게이션 바 컨테이너 */
    .top-nav {
        position: sticky;
        top: 0;
        z-index: 1000;
        background: #ffffff;
        padding: 0;
        border-bottom: 1px solid #e5e5e5;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    
    /* 네비게이션 내부 래퍼 */
    .nav-wrapper {
        max-width: 1400px;
        margin: 0 auto;
        padding: 0 40px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        height: 70px;
    }
    
    /* 로고 영역 */
    .nav-logo {
        font-size: 24px;
        font-weight: 700;
        color: #000;
        letter-spacing: -0.5px;
    }
    
    /* 메뉴 컨테이너 - 균등 배치 */
    .nav-menu-container {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 40px;
        flex: 1;
    }
    
    /* Streamlit columns 간격 제거 */
    [data-testid="column"] {
        padding: 0 !important;
    }
    
    /* Streamlit 버튼 스타일 완전 재정의 */
    .stButton {
        margin: 0 !important;
    }
    
    .stButton > button {
        background: transparent !important;
        border: none !important;
        color: #333 !important;
        font-size: 15px !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        position: relative !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        height: auto !important;
        min-height: auto !important;
        white-space: nowrap !important;
        width: 100% !important;
        text-align: center !important;
    }
    
    .stButton > button:hover {
        background: transparent !important;
        color: #000 !important;
        box-shadow: none !important;
        border: none !important;
    }
    
    .stButton > button:active,
    .stButton > button:focus {
        background: transparent !important;
        color: #000 !important;
        box-shadow: none !important;
        border: none !important;
    }
    
    /* 호버 언더라인 효과 */
    .stButton > button::after {
        content: '';
        position: absolute;
        bottom: 4px;
        left: 50%;
        transform: translateX(-50%);
        width: 0;
        height: 2px;
        background: #000;
        transition: width 0.3s ease;
    }
    
    .stButton > button:hover::after {
        width: 80%;
    }
    
    /* 다크모드 대응 */
    @media (prefers-color-scheme: dark) {
        .top-nav {
            background: #0E1117;
            border-bottom: 1px solid #2d2d2d;
        }
        
        .nav-logo {
            color: #fff;
        }
        
        .stButton > button {
            color: #e0e0e0 !important;
        }
        
        .stButton > button:hover {
            color: #fff !important;
        }
        
        .stButton > button::after {
            background: #fff;
        }
    }
    
    /* 반응형 디자인 */
    @media (max-width: 768px) {
        .nav-wrapper {
            padding: 0 20px;
            height: 60px;
        }
        
        .nav-logo {
            font-size: 20px;
        }
        
        .nav-menu-container {
            gap: 20px;
        }
        
        .stButton > button {
            font-size: 14px !important;
            padding: 6px 12px !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # 네비게이션 바 구조
    st.markdown('<div class="top-nav">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.markdown('<div class="nav-logo">⚡ 3TEAM</div>', unsafe_allow_html=True)
    
    with col2:
        # 메뉴 버튼들을 균등하게 배치 (각 버튼이 같은 너비)
        menu_cols = st.columns([3,1,1,1,1,1])
        
        with menu_cols[1]:
            if st.button("Home", key="nav_home"):
                st.switch_page("./Home.py")
        
        with menu_cols[2]:
            if st.button("Overview", key="nav_overview"):
                st.switch_page("pages/1_Overview.py")
        
        with menu_cols[3]:
            if st.button("Model", key="nav_model"):
                st.switch_page("pages/2_Model_Compare.py")
        
        with menu_cols[4]:
            if st.button("Report", key="nav_report"):
                st.switch_page("pages/3_Report_Download.py")
        
        with menu_cols[5]:
            if st.button("Q&A", key="nav_qna"):
                st.switch_page("pages/4_FAQ_QnA.py")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========= 1_Overview.py =========

def overview_ui():
    st.markdown("""
    <style>
        /* 섹션 타이틀 공통 */
        .section-title {
            font-size: 1.4rem;
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 0.8rem;
            border-left: 5px solid #2563eb;
            padding-left: 10px;
        }

        /* Challenge 카드 스타일 */
        .challenge-card {
            background-color: white;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 15px;
            height: 100%;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .icon-box { font-size: 1.8rem; margin-bottom: 8px; }
        .card-title { font-size: 1.1rem; font-weight: 700; margin-bottom: 8px; color: #111; }
        .card-text { font-size: 0.9rem; color: #555; line-height: 1.4; word-break: keep-all; }

        /* Solution 스타일 */
        .solution-box {
            background-color: #f8fafc;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #e2e8f0;
        }
        .solution-item { display: flex; gap: 12px; margin-bottom: 15px; }
        .solution-number {
            background-color: #2563eb; color: white;
            width: 24px; height: 24px; border-radius: 50%;
            text-align: center; line-height: 24px; font-weight: bold; font-size: 0.8rem;
            flex-shrink: 0; margin-top: 3px;
        }
        .solution-title { font-weight: 700; color: #1e293b; font-size: 1rem; margin-bottom: 4px; }
        .solution-text { font-size: 0.9rem; color: #475569; line-height: 1.4; word-break: keep-all; }
        
        /* Tech Badge */
        .tech-badge {
            background-color: #e0e7ff; color: #3730a3;
            padding: 4px 8px; border-radius: 6px;
            font-size: 0.75rem; font-weight: 600; margin-right: 5px;
        }

        /* 🏆 Impact - 메트릭 카드 (상단 3개) */
        .metric-card {
            background-color: white;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 12px; /* 패딩 축소 */
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .metric-value {
            font-size: 1.6rem; /* 글씨 크기 축소 */
            font-weight: 800;
            color: #2563eb;
            margin-bottom: 4px;
        }
        .metric-label {
            font-size: 0.85rem;
            color: #6b7280;
            font-weight: 600;
            word-break: keep-all;
        }

        /* Impact - 하단 노트 */
        .impact-note {
            margin-top: 15px;
            padding: 8px;
            background-color: #f8fafc;
            border-radius: 8px;
            font-size: 0.8rem;
            color: #475569;
            text-align: center;
            border: 1px dashed #cbd5e1;
        }

        /* 📊 프로젝트 핵심 지표 (하단 박스) - 높이 축소 핵심 */
        .metrics-box {
            background-color: #f9fafb;
            border-radius: 10px;
            padding: 10px 35px; /* 상하 패딩을 줄여서 높이 축소 */
            border: 1px solid #f3f4f6;
            margin-top: 0px;
        }
        .metrics-title {
            font-size: 1rem;
            font-weight: 700;
            color: #374151;
            margin-bottom: 5px; /* 간격 축소 */
            text-align: center;
        }
        .metric-item {
            text-align: center;
            padding: 0px; /* 불필요한 패딩 제거 */
        }
        .metric-item-value {
            font-size: 1.2rem; /* 글씨 크기 축소 */
            font-weight: 800;
            margin-bottom: 2px;
        }
        .metric-item-label {
            font-size: 0.75rem;
            color: #6b7280;
            font-weight: 500;
            word-break: keep-all;
        }
    </style>
    """, unsafe_allow_html=True)

# =========== 2_Model_Compare.py ===========
def model_ui():
    st.markdown("""
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 2rem; }
        .metric-label { font-size: 14px; color: #666; margin-bottom: 2px; font-weight: 600; }
        .metric-value { font-size: 26px; color: #333; font-weight: 800; }
        .metric-sub { font-size: 12px; color: #888; margin-top: 2px; }
        .cutoff-info {
            font-size: 13px; color: #444; background-color: #f8f9fa;
            padding: 5px 10px; border-radius: 6px; border: 1px solid #eee;
            margin-top: -5px; display: inline-block;
        }
        .compare-header { font-size: 20px; font-weight: bold; color: #333; text-align: center; margin-bottom: 10px;}
        .section-header { font-size: 16px; font-weight: bold; margin-bottom: 5px; }
        
        /* VS 배지 스타일 */
        .vs-badge-large { 
            font-size: 24px; font-weight: 900; color: #FF4B4B; 
            text-align: center; padding: 20px 0; margin-top: 100px;
        }
    </style>
    """, unsafe_allow_html=True)

# hover + active
def apply_tooltip_style():
    """툴팁 공통 CSS (너비 확대 및 위쪽 표시로 변경)"""
    st.markdown("""
    <style>
    /* 툴팁 트리거 텍스트 스타일 */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help; /* 커서를 물음표나 손가락으로 변경 */
        font-weight: 700;
        color: #2563eb;
        border-bottom: 1px dashed #2563eb; /* 툴팁이 있다는 시각적 힌트 추가 */
    }

    /* 툴팁 박스 스타일 */
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 320px; /* 너비를 넓혀서 개행 문제 해결 (220px -> 320px) */
        background-color: #1f2937; /* 다크 그레이 배경 */
        color: #fff;
        text-align: left;
        border-radius: 8px;
        padding: 12px 16px;
        
        /* 위치 조정: 텍스트의 '위쪽'에 뜨도록 설정 */
        position: absolute;
        z-index: 9999;
        bottom: 135%; /* 텍스트 위로 띄움 (Top -> Bottom 변경) */
        left: 50%;
        transform: translateX(-50%); /* 중앙 정렬 */
        
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.9rem;
        line-height: 1.5;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        font-weight: 400; /* 본문 폰트 두께 조절 */
    }

    /* 툴팁 화살표 (아래쪽을 가리키도록 변경) */
    .tooltip .tooltiptext::after {
        content: "";
        position: absolute;
        top: 100%; /* 툴팁 박스 바로 아래 */
        left: 50%;
        margin-left: -6px;
        border-width: 6px;
        border-style: solid;
        border-color: #1f2937 transparent transparent transparent; /* 위쪽 색상만 지정 */
    }

    /* 호버 시 표시 */
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)


MODEL_TOOLTIP = {
    # 머신러닝 모델 
    "Logistic Regression": """
        <b>📉 로지스틱 회귀 (Logistic Regression)</b><br>
        데이터의 선형적 관계를 기반으로 이진 분류를 수행하는 가장 기초적인 모델입니다.
    """,

    "Random Forest": """
        <b>🌲 랜덤 포레스트 (Random Forest)</b><br>
        여러 개의 결정 트리(Decision Tree)를 만들고 다수결(Bagging)로 예측하는 앙상블 모델입니다.
    """,

    "XGBoost": """
        <b>🚀 XGBoost</b><br>
        이전 트리의 오차를 순차적으로 학습(Boosting)하여 성능을 극대화한 강력한 모델입니다.
    """,

    "LightGBM": """
        <b>⚡ LightGBM</b><br>
        리프 중심(Leaf-wise) 성장 방식을 사용하여, 대용량 데이터도 빠르고 효율적으로 학습하는 모델입니다.
    """,

    "HistGradientBoosting": """
        <b>📊 HistGradientBoosting (HGB)</b><br>
        데이터를 히스토그램(구간)으로 나누어 처리 속도를 획기적으로 높인 효율적인 부스팅 모델입니다.
    """,

    "Decision Tree": """
        <b>🌳 의사결정 나무 (Decision Tree)</b><br>
        데이터의 특징을 스무고개 하듯 단계적으로 분할하여 분류하는 직관적인 모델입니다.
    """,

    "ExtraTrees": """
        <b>🌲 엑스트라 트리 (ExtraTrees)</b><br>
        랜덤 포레스트보다 무작위성을 더 높여 과적합을 방지하고 속도를 높인 앙상블 모델입니다.
    """,

    # 딥러닝 모델
    "DNN (MLP)": """
        <b>🧠 DNN (Multi-Layer Perceptron)</b><br>
        입력층과 출력층 사이에 여러 은닉층을 두어 복잡한 비선형 관계를 학습하는 심층 신경망입니다.
    """,

    "TabNet": """
        <b>📑 TabNet</b><br>
        트리 모델의 장점(특징 선택)과 딥러닝의 장점(표현 학습)을 결합한 정형 데이터 특화 모델입니다.
    """,

    "Wide & Deep": """
        <b>🌐 Wide & Deep</b><br>
        암기(Wide)와 일반화(Deep)를 동시에 수행하여 추천 시스템 등에 효과적인 하이브리드 모델입니다.
    """,
    
    # (기존 코드에 있던 이름 대응)
    "Baseline MLP": """
        <b>🧠 Baseline MLP</b><br>
        가장 기본적인 형태의 심층 신경망으로, 딥러닝 성능 비교의 기준점이 되는 모델입니다.
    """,

    "Deep ResNet": """
        <b>🔗 Deep ResNet</b><br>
        잔차 연결(Skip Connection)을 통해 층을 매우 깊게 쌓아도 학습이 잘 되도록 설계된 모델입니다.
    """
}


def model_tooltip(model_name: str, color: str = "#2563eb"):
    desc = MODEL_TOOLTIP.get(model_name, "설명이 없습니다.")

    return f"""
    <span class="tooltip" style="color: {color}; border-bottom: 1px dashed {color};">
        {model_name}
        <span class="tooltiptext">{desc}</span>
    </span>
    """


    # st.markdown(
    #     f"""
    #     <span class="tooltip">{model_name}
    #         <span class="tooltiptext">{desc}</span>
    #     </span>
    #     """,
    #     unsafe_allow_html=True
    # )



# ============ 4_FAQ_QnA.py ============

# CSS 스타일링
def QnA_ui():
    st.markdown("""
    <style>
        /* 전체 배경 */
        .main {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        }
        
        /* Hero 섹션 */
        .faq-hero {
            background: linear-gradient(135deg, #1e293b 0%, #475569 100%);
            padding: 50px 40px;
            border-radius: 20px;
            margin-bottom: 40px;
            color: white;
            text-align: center;
        }
        
        .faq-hero-title {
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 15px;
        }
        
        .faq-hero-subtitle {
            font-size: 1.15rem;
            color: #cbd5e1;
            max-width: 700px;
            margin: 0 auto;
        }
        
        /* 카테고리 탭 */
        .category-tabs {
            display: flex;
            gap: 15px;
            margin-bottom: 40px;
            flex-wrap: wrap;
            justify-content: center;
        }
        
        .category-tab {
            background: white;
            border: 2px solid #e2e8f0;
            padding: 12px 30px;
            border-radius: 30px;
            font-weight: 600;
            color: #64748b;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .category-tab:hover {
            border-color: #3b82f6;
            color: #3b82f6;
            transform: translateY(-2px);
        }
        
        .category-tab.active {
            background: #3b82f6;
            border-color: #3b82f6;
            color: white;
        }
        
        /* 검색 바 */
        .search-container {
            max-width: 700px;
            margin: 0 auto 50px auto;
        }
        
        /* FAQ 카드 스타일 - Streamlit expander 커스텀 */
        .streamlit-expanderHeader {
            background: white !important;
            border-radius: 12px !important;
            border: 2px solid #e2e8f0 !important;
            padding: 20px 25px !important;
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            color: #1e293b !important;
            transition: all 0.3s ease !important;
            margin-bottom: 15px !important;
        }
        
        .streamlit-expanderHeader:hover {
            border-color: #3b82f6 !important;
            background: #f8fafc !important;
        }
        
        [data-testid="stExpander"] {
            background: transparent !important;
            border: none !important;
        }
        
        .streamlit-expanderContent {
            background: white !important;
            border: 2px solid #e2e8f0 !important;
            border-top: none !important;
            border-radius: 0 0 12px 12px !important;
            padding: 25px !important;
            margin-top: -15px !important;
        }
        
        /* 카테고리 섹션 타이틀 */
        .category-title {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1e293b;
            margin: 50px 0 25px 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .category-badge {
            display: inline-block;
            background: #3b82f6;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        
        /* 코드 블록 스타일 */
        code {
            background: #f1f5f9 !important;
            color: #1e293b !important;
            padding: 2px 8px !important;
            border-radius: 4px !important;
            font-size: 0.9em !important;
        }
        
        /* 테이블 스타일 */
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        
        th {
            background: #f1f5f9;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #1e293b;
        }
        
        td {
            padding: 12px;
            border-bottom: 1px solid #e2e8f0;
        }
        
        /* Contact CTA */
        .contact-cta {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            padding: 40px;
            border-radius: 20px;
            color: white;
            text-align: center;
            margin-top: 60px;
        }
        
        .contact-cta h3 {
            font-size: 1.8rem;
            margin-bottom: 15px;
        }
        
        .contact-cta p {
            font-size: 1.1rem;
            color: #bfdbfe;
            margin-bottom: 25px;
        }
        
        /* 통계 박스 */
        .stat-box {
            background: white;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #3b82f6;
            margin-bottom: 5px;
        }
        
        .stat-label {
            font-size: 0.9rem;
            color: #64748b;
        }
    </style>
    """, unsafe_allow_html=True)