import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from ocr_utils import extract_number, extract_height_weight_pair, clean_feature_value, calculate_age_group,detect_text_from_image
from model_predict import predict_proba
from configs import COEFFICIENTS, NORMALIZATION_RULES
from food_recommender import (
    normalize_health_inputs, calculate_risk_score, calculate_bmr_mifflin,adjust_dash_standard,
    compute_metabolic_score, compute_health_index, recommend_foods, classify_risk
)
from health_report import (
    display_user_profile,
    display_risk_scores,
    display_contributing_factors,
    display_management_tip
)

DAILY_VALUES = {
    "나트륨(mg)": 2000, "탄수화물(g)": 324, "당류(g)": 100,
    "지방(g)": 54, "포화지방산(g)": 15, "콜레스테롤(mg)": 300,
    "단백질(g)": 55, "식이섬유(g)": 25, "칼륨(mg)": 3500, "칼슘(mg)": 700
}

NUTRIENT_INFO_POPOVER = {
    "나트륨(mg)": "나트륨은 체액 균형에 필요하지만, 과다 섭취 시 혈압 상승의 주요 원인이 됩니다. DASH 다이어트에서는 엄격한 제한을 권장합니다.",
    "칼륨(mg)": "칼륨은 나트륨 배출을 돕고 혈관을 이완시켜 혈압을 낮추는 데 도움이 됩니다. DASH 다이어트에서는 충분한 섭취를 강조합니다.",
    "식이섬유(g)": "식이섬유는 혈중 콜레스테롤 수치를 개선하고 혈당 조절에 도움을 주어 혈압 관리에 긍정적인 영향을 미칩니다. DASH 다이어트의 중요한 요소입니다.",
    "포화지방산(g)": "포화지방산의 과다 섭취는 혈중 LDL 콜레스테롤 수치를 높여 심혈관 질환 위험을 증가시킬 수 있습니다. DASH 다이어트에서는 섭취 제한을 권장합니다.",
    "콜레스테롤(mg)": "DASH 다이어트에서는 콜레스테롤 섭취를 제한할 것을 권장합니다.",
    "칼슘(mg)": "칼슘은 정상 혈압 유지에 기여하며, DASH 다이어트에서 권장되는 미네랄입니다.",
    "마그네슘(mg)": "마그네슘은 혈압 조절에 도움을 줄 수 있으며, DASH 다이어트에서 권장되는 미네랄입니다. (food_df에 '마그네슘(mg)' 컬럼이 있다면 추가)"
}

# ✅ 초기화는 항상 최상단
st.set_page_config(page_title="HiNavi - OCR 건강진단", layout="centered")

def get_styled_score_display(score):
    score = float(score)  # 점수가 문자열일 경우를 대비해 float으로 변환

    # 점수 범위에 따른 배경색 및 텍스트 색상 설정
    if score >= 85:
        background_color = "#28a745"  # 매우 좋음 (진한 녹색)
        text_color = "white"
    elif score >= 70:
        background_color = "#20c997"  # 좋음 (녹색 계열)
        text_color = "white"
    elif score >= 50:
        background_color = "#ffc107"  # 보통 (노란색)
        text_color = "black" # 노란색 배경에는 검은색 텍스트가 잘 보임
    else:
        background_color = "#dc3545"  # 개선 필요 (빨간색)
        text_color = "white"

    # HTML로 스타일링된 점수 태그 생성
    styled_score_html = f"""
    <span style="
        background-color: {background_color};
        color: {text_color};
        padding: 3px 8px; /* 패딩 조절 */
        border-radius: 10px; /* 둥근 모서리 */
        font-weight: bold;
        font-size: 0.9em; /* 폰트 크기 조절 */
    ">
        {score:.1f}점
    </span>
    """
    return styled_score_html

# ✅ 배경색 및 버튼 스타일만 가볍게 유지
st.markdown("""
<style>
/* 전체 앱 배경색 */
.stApp { background-color: #313332; }

/* 버튼 스타일 - 예측 실행하기 버튼에 적용될 기본 스타일 */
.stButton > button {
    background-color: #27AE60; /* 녹색 계열 */
    color: white;
    padding: 15px 50px;
    border: none;
    border-radius: 10px;
    font-size: 20px;
    font-weight: bold;
    cursor: pointer;
    transition: background-color 0.3s ease; /* 호버 효과를 위한 transition */
}
.stButton > button:hover {
    background-color: #1F8F4F; /* 호버 시 더 진한 녹색 */
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* 그림자 효과 추가 */
}

/* 텍스트 입력 필드 및 숫자 입력 필드 스타일 */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stNumberInput > div > div > input {
    background-color: #444444; /* 어두운 회색 배경 */
    color: white;
    border: 1px solid #555555;
    border-radius: 5px;
    padding: 10px;
}

/* Selectbox 스타일 */
.stSelectbox > div > div > div > div {
    background-color: #444444;
    color: white;
    border: 1px solid #555555;
    border-radius: 5px;
}
.stSelectbox > div > div > div > div > span { /* 선택된 항목 텍스트 색상 */
    color: white;
}
.stSelectbox > div > div > div > div > div > div { /* 드롭다운 화살표 색상 */
    color: #CCCCCC;
}

/* 파일 업로더 스타일 */
.stFileUploader {
    background-color: #444444;
    border: 1px dashed #555555;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    color: #CCCCCC;
}
.stFileUploader > div > div > button { /* Browse files 버튼 */
    background-color: #555555;
    color: white;
    border-radius: 5px;
    padding: 8px 15px;
}
.stFileUploader > div > div > button:hover {
    background-color: #666666;
}

/* 정보 박스 (st.info) */
.stAlert {
    background-color: #3C4B64; /* 좀 더 차분한 파란색 계열 */
    color: white;
    border-radius: 8px;
    padding: 15px;
    border: none;
}
.logo-container {
    background-image: url("https://i.imgur.com/rClP4Ah.png");
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
    height: 150px; /* 로고 이미지 높이 조절 */
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)


# ✅ 로고 이미지 삽입 - 상단에 고정
st.markdown("""
<div class='logo-container'></div>
""", unsafe_allow_html=True)

# ✅ 메인 타이틀 및 부제목
st.markdown("""
<h1 style="color:#2ECC71; font-size:52px; text-align:center; margin-bottom:10px;">HiNavi</h1>
<h3 style="color:#F39C12; font-size:26px; text-align:center; margin-bottom:30px;">OCR 기반 맞춤형 건강진단 · 식품추천</h3>
<p style="color:#BBBBBB; font-size:18px; text-align:center; line-height:1.5;">
    건강검진 결과를 OCR로 간편하게 등록하고, 나만의 식단 추천을 받아보세요.<br>
    간편한 건강 관리를 HiNavi와 함께 시작해보세요!
</p>
<br>
""", unsafe_allow_html=True)

# ✅ 데이터 로딩
@st.cache_data
def load_food_data():
    return pd.read_csv("data/raw/combined_filterd_fd.csv")
food_df = load_food_data()

# ✅ 세션 상태 초기화 (페이지 + 히스토리 스택)
if 'page' not in st.session_state:
    st.session_state.page = 'input'
if 'history' not in st.session_state:
    st.session_state.history = []

# ✅ 페이지 이동/뒤로가기 헬퍼함수
def navigate_to(next_page):
    st.session_state.history.append(st.session_state.page)
    st.session_state.page = next_page
    st.rerun()

def go_back():
    if st.session_state.history:
        st.session_state.page = st.session_state.history.pop()
        st.rerun()

# ✅ 게이지 차트 함수
def hypertension_gauge(proba):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = proba * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "고혈압 위험도", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#2E86C1"},
            'steps': [
                {'range': [0, 40], 'color': "#2ecc71"},
                {'range': [40, 65], 'color': "#f1c40f"},
                {'range': [65, 85], 'color': "#e67e22"},
                {'range': [85, 100], 'color': "#e74c3c"},
            ],
        }
    ))
    st.plotly_chart(fig)

# ✅ OCR 입력 단계
if st.session_state.page == 'input':
    st.markdown("<h2 style='color:#7CB9E8; text-align:center;'>🙋‍♀️ 사용자 정보 입력</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True) # 여백 추가

    col_age, col_sex = st.columns(2)
    with col_age:
        age = st.number_input("나이", 10, 100, 30, help="만 나이를 입력해주세요.", key="input_age")
        real_age = age
    with col_sex:
        sex = st.selectbox("성별", ["남성", "여성"], help="성별을 선택해주세요.", key="input_sex")

    st.markdown("<br>", unsafe_allow_html=True) # 여백 추가
    st.markdown("<h2 style='color:#7CB9E8; text-align:center;'>📸 건강검진 이미지 업로드</h2>", unsafe_allow_html=True)
    st.caption("건강검진 결과 이미지를 업로드하시면 OCR로 텍스트를 자동으로 인식합니다.")
    uploaded_file = st.file_uploader("건강검진 결과 이미지 (JPG/PNG)", type=["jpg", "png", "jpeg"], key="uploaded_file")

    ocr_text_from_image = "" # 이미지로부터 추출된 텍스트를 저장할 변수

    if uploaded_file is not None: # 파일이 업로드되었다면
        st.image(uploaded_file, caption="업로드된 이미지", use_container_width=True)
        # OCR 실행
        try:
            with st.spinner("이미지에서 텍스트를 추출 중입니다..."):
                ocr_text_from_image = detect_text_from_image(uploaded_file)
            if "CLOVA OCR 클라이언트를 초기화할 수 없습니다" in ocr_text_from_image or "오류" in ocr_text_from_image:
                 st.error(f"OCR 오류: {ocr_text_from_image}")
                 ocr_text_from_image = ""
            else:
                 st.success("이미지에서 텍스트를 성공적으로 추출했습니다!")
        except Exception as e:
            st.error(f"OCR 처리 중 오류가 발생했습니다: {e}")
            ocr_text_from_image = ""
    else:
        st.info("여기에 건강검진 결과를 스캔한 이미지 파일을 업로드해주세요.")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- OCR 파싱 결과 확인 섹션 ---
    st.markdown("<h2 style='color:#7CB9E8; text-align:center;'>✨ 추출된 건강 지표 확인</h2>", unsafe_allow_html=True)
    st.info("OCR을 통해 이미지에서 추출된 주요 건강 지표입니다. 예측 정확도를 위해 올바르게 추출되었는지 확인해주세요.")

    # --- '여기를 눌러 수정하세요' 섹션 (Expander) ---
    # 파싱에 사용할 텍스트를 결정하는 위젯 (항상 렌더링되면서 값은 Streamlit이 관리)
    # edited_ocr_text는 사용자가 입력하는 텍스트 영역의 현재 값을 직접 가리킴
    with st.expander("💡 추출된 정보가 정확하지 않다면 여기를 클릭하여 직접 입력/수정하세요.", expanded=False):
        edited_ocr_text_value = st.text_area("건강검진 결과 텍스트를 입력하거나 수정해주세요.",
                                            value=ocr_text_from_image, # OCR 결과가 기본값
                                            height=120, # 높이를 더 줄임
                                            help="예: 키 170 몸무게 65 공복혈당 90 혈색소 14 감마지티피 30 중성지방 100 HDL 50",
                                            key="edited_ocr_text_area") # 고유 key 추가
        
        # 만약 사용자가 텍스트 에어리어의 값을 수정했다면, 그 수정된 값을 파싱에 사용.
        # Streamlit은 위젯의 value가 변경되면 자동으로 앱을 리런하므로,
        # 이 시점에는 edited_ocr_text_value가 최신 값을 가지고 있습니다.
        # 따라서, ocr_text_to_parse는 edited_ocr_text_value를 참조하게 됩니다.

    # ocr_text_to_parse 변수 결정:
    # Expander가 열려있고 (key="edited_ocr_text_area" 위젯이 렌더링되었고),
    # 그 위젯의 현재 값이 초기값(ocr_text_from_image)과 다르다면 edited_ocr_text_value를 사용.
    # 그렇지 않다면 (Expander가 닫혀있거나 수정되지 않았다면) ocr_text_from_image를 사용.
    # Streamlit의 `st.session_state`를 사용하여 위젯의 현재 상태를 확인하는 것이 가장 정확합니다.
    if "edited_ocr_text_area" in st.session_state and st.session_state.edited_ocr_text_area != ocr_text_from_image:
        ocr_text_to_parse = st.session_state.edited_ocr_text_area
    else:
        ocr_text_to_parse = ocr_text_from_image


    # 실시간 파싱 로직 (이제 ocr_text_to_parse 변수를 사용합니다.)
    height_cm, weight_kg = extract_height_weight_pair(ocr_text_to_parse)
    if height_cm is None:
        height_cm = extract_number(["키", "신장"], ocr_text_to_parse)
    if weight_kg is None:
        weight_kg = extract_number(["몸무게", "체중"], ocr_text_to_parse)

    bmi = 0
    if height_cm and weight_kg:
        bmi = round(weight_kg / ((height_cm / 100) ** 2), 2)
        print(f"DEBUG: BMI 계산됨: {bmi} (키: {height_cm}, 몸무게: {weight_kg})")

    features = {
        '연령대코드(5세단위)': calculate_age_group(age),
        '식전혈당(공복혈당)': clean_feature_value('식전혈당(공복혈당)', extract_number(["공복혈당", "혈당"], ocr_text_to_parse)),
        '혈색소': clean_feature_value('혈색소', extract_number(["혈색소"], ocr_text_to_parse)),
        '감마지티피': clean_feature_value('감마지티피', extract_number(["감마지티피", "GTP", "y-GTP"], ocr_text_to_parse)),
        '트리글리세라이드': clean_feature_value('트리글리세라이드', extract_number(["중성지방",'트리글리세라이드'], ocr_text_to_parse)),
        'HDL콜레스테롤': clean_feature_value('HDL콜레스테롤', extract_number(["고밀도 콜레스테롤",'HDL콜레스테롤', "HDL"], ocr_text_to_parse)),
        'bmi': bmi
    }

    display_names = { # 기존 정의된 display_names 사용
        '연령대코드(5세단위)': '나이 (연령대 5세단위)',
        '식전혈당(공복혈당)': '공복 혈당',
        '혈색소': '혈색소',
        '감마지티피': '감마지티피',
        '트리글리세라이드': '트리글리세라이드 (중성지방)',
        'HDL콜레스테롤': 'HDL 콜레스테롤',
        'bmi': 'BMI'
    }

    col_left, col_right = st.columns(2)
    missing_or_invalid_display = []

    for i, (key, value) in enumerate(features.items()):
        display_name = display_names.get(key, key)
        is_valid = True
        if value is None or (isinstance(value, (float, np.number)) and (np.isnan(value) or np.isinf(value))):
            display_value = "정보 없음"
            is_valid = False
        elif key == 'bmi' and value == 0 and (not height_cm or not weight_kg):
            display_value = f"{value:.1f} (키/몸무게 부족)"
            is_valid = False
        elif isinstance(value, (int, float, np.number)):
            display_value = f"{value:.1f}"
        else:
            display_value = str(value)

        target_col = col_left if i % 2 == 0 else col_right
        with target_col:
            st.markdown(f"**{display_name}:** {display_value}")

        if not is_valid:
            missing_or_invalid_display.append(display_name)

    # 최종 파싱되지 않았거나 잘못된 값들에 대한 경고 메시지
    if missing_or_invalid_display:
        st.warning(f"일부 건강 지표가 추출되지 않았거나 유효하지 않습니다: {', '.join(missing_or_invalid_display)}. "
                   f"상단의 '추출된 정보가 정확하지 않다면 여기를 클릭하여 직접 입력/수정하세요.'를 열어 텍스트를 직접 입력하거나 수정해주세요.")

    st.markdown("<br>", unsafe_allow_html=True)
    col_empty1, col_button, col_empty2 = st.columns([1, 2, 1]) # 버튼 중앙 정렬을 위한 컬럼 분할
    with col_button:
        if st.button("✨ 예측 실행하기", use_container_width=True):
            required_features_for_predict = [
                '연령대코드(5세단위)', '식전혈당(공복혈당)', '혈색소',
                '감마지티피', 'bmi', '트리글리세라이드', 'HDL콜레스테롤'
            ]

            missing_for_predict = []
            for f_key in required_features_for_predict:
                val = features.get(f_key)
                if val is None or (isinstance(val, (float, np.number)) and (np.isnan(val) or np.isinf(val))):
                    if f_key == 'bmi' and val == 0:
                        continue
                    missing_for_predict.append(display_names.get(f_key, f_key))

            if missing_for_predict:
                st.error(f"예측에 필요한 다음 정보가 부족하거나 유효하지 않습니다: {', '.join(missing_for_predict)}. "
                         "상단의 '추출된 정보가 정확하지 않다면 여기를 클릭하여 직접 입력/수정하세요.'를 열어 정보를 직접 입력하거나 수정해주세요.")
                st.stop()

            try:
                proba = predict_proba(features)
            except Exception as e:
                st.error(f"예측 중 오류가 발생했습니다. 입력값을 확인해주세요. 오류: {e}")
                st.stop()

            risk_level = classify_risk(proba)

            st.session_state.update({
                "features": features, "age": age, "sex": sex,
                "height_cm": height_cm, "weight_kg": weight_kg, "bmi": bmi,
                "proba": proba, "risk_level": risk_level
            })
            navigate_to("predict_result")

# ✅ 2단계: 예측 결과
elif st.session_state.page == 'predict_result':
    st.title("🩺 나의 고혈압 위험도 분석 결과") # 페이지 제목 변경 제안

    # 위험도 게이지 차트 표시
    hypertension_gauge(st.session_state.proba)

    # 위험도 등급 및 상세 설명
    risk_level = st.session_state.risk_level
    proba_percent = st.session_state.proba * 100

    st.markdown(f"<h3 style='text-align: center;'>현재 회원님의 고혈압 위험도는 <span style='color: #F39C12;'>{proba_percent:.1f}%</span>로, <span style='color: #F39C12;'>'{risk_level}'</span> 단계입니다.</h3>", unsafe_allow_html=True)

    # 등급별 상세 설명 추가
    if risk_level == '양호':
        st.success("""
        **✨ 양호 (Good)**

        현재 고혈압 위험도가 낮은 상태입니다. 건강한 생활 습관을 잘 유지하고 계시는군요!
        계속해서 균형 잡힌 식단과 규칙적인 운동을 실천하며 현재 상태를 지켜나가세요.
        """)
    elif risk_level == '주의':
        st.warning("""
        **⚠️ 주의 (Caution)**

        고혈압 발병 위험에 주의가 필요한 단계입니다. 아직 질병 상태는 아니지만, 방심하면 위험 단계로 진행될 수 있어요.
        생활 습관 개선(식단 조절, 운동 시작 등)을 통해 적극적으로 관리하는 것이 좋습니다.
        아래 '상세 건강 분석 보기'를 통해 어떤 요인에 주의해야 할지 확인해보세요.
        """)
    elif risk_level == '위험':
        st.error("""
        **🚨 위험 (Warning)**

        고혈압 발병 위험도가 높은 상태로, 적극적인 관리가 필요합니다.
        식습관 개선, 규칙적인 운동, 스트레스 관리 등 생활 전반의 변화가 중요합니다.
        의료 전문가와 상담하여 개인 맞춤형 관리 계획을 세우는 것을 권장합니다.
        '상세 건강 분석 보기'에서 위험 요인을 자세히 살펴보세요.
        """)
    elif risk_level == '고위험':
        st.error("""
        **🛑 고위험 (High Risk)**

        고혈압 발병 위험도가 매우 높은 상태입니다. 즉각적인 생활 습관 개선과 함께 반드시 의료 전문가의 진료 및 상담이 필요합니다.
        합병증 예방을 위한 적극적인 관리가 중요하니, '상세 건강 분석 보기'를 통해 위험 요인을 확인하고 전문가와 상담하세요.
        """)

    st.markdown("---") # 구분선 추가

    # 네비게이션 버튼 (가운데 정렬 및 버튼 텍스트 수정 제안)
    col1, col2, col3 = st.columns([1, 2, 1]) # 중앙 컬럼 너비 확장

    with col1:
        if st.button("◀ 이전으로 돌아가기"): # 버튼 텍스트 변경 제안
            go_back()

    with col3:
        if st.button("상세 건강 분석 보기 🔬"): # 버튼 텍스트 및 아이콘 추가 제안
            navigate_to("report")

# ✅ 3단계: 건강검진 분석
elif st.session_state.page == 'report':
    st.title("건강검진 데이터 분석 결과")

    # 세션 상태에서 필요한 값들 가져오기
    age = st.session_state.age
    sex = st.session_state.sex
    height_cm = st.session_state.height_cm
    weight_kg = st.session_state.weight_kg
    bmi = st.session_state.bmi

    proba = st.session_state.proba
    risk_level = st.session_state.risk_level

    features = st.session_state.features # 'input' 페이지에서 저장된 features

    # health_report.py에서 사용할 값들 계산
    normalized = normalize_health_inputs(features, NORMALIZATION_RULES)
    risk_score = calculate_risk_score(normalized, COEFFICIENTS)

    contributions = {
        key: round((normalized.get(key, 0) or 0) * COEFFICIENTS.get(key, 0), 4)
        for key in COEFFICIENTS
    }

    # ❗ BMR 계산: st.session_state.bmr을 읽는 대신 직접 계산합니다.
    bmr_value = calculate_bmr_mifflin(sex, age, height_cm, weight_kg)

    # 계산된 bmr_value (로컬 변수)를 사용합니다.
    metabolic_score = compute_metabolic_score(bmr_value, bmi)
    health_index = compute_health_index(risk_score, metabolic_score)

    # health_report.py의 함수들 호출
    display_user_profile(age, sex, height_cm, weight_kg, bmi)
    st.markdown("---")
    display_risk_scores(proba, risk_level, risk_score, metabolic_score, health_index)
    st.markdown("---")
    display_contributing_factors(contributions, bmi, age)
    st.markdown("---")
    display_management_tip(contributions, bmi) # bmi 값 전달
    st.markdown("---")

    # 다음 페이지('recommend')에서 사용하기 위해 세션 상태에 저장
    st.session_state.health_index = health_index
    st.session_state.bmr = bmr_value # 계산된 bmr_value를 세션 상태에 저장

    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("◀ 이전으로 돌아가기"):
            go_back()
    with col_nav2:
        if st.button("✅ 맞춤 식품 추천 확인하기"):
            with st.spinner("맞춤 식품을 추천 중입니다... 잠시만 기다려 주세요."):
                INITIAL_ITEMS_TO_SHOW = 5
                INCREMENT_BY = 5
                MAX_RECOMMENDATIONS_TO_FETCH = 100
                LOWEST_ITEMS_TO_SHOW = 5
                try:
                    top_recs, lowest_recs = recommend_foods( # <-- 이 부분 수정
                        st.session_state.bmr,
                        st.session_state.health_index,
                        food_df,
                        top_n=MAX_RECOMMENDATIONS_TO_FETCH,
                        lowest_n=LOWEST_ITEMS_TO_SHOW
                        # 좋아요/싫어요 기능은 현재 사용하지 않으므로 인자 전달 제거
                        # liked_foods=None,
                        # disliked_foods=None
                    )
                    st.session_state.all_recommendations_cache = top_recs # 상위 추천 식품 저장
                    st.session_state.lowest_foods_cache = lowest_recs
                    st.session_state.num_items_currently_showing = INITIAL_ITEMS_TO_SHOW
                    st.session_state.increment_amount = INCREMENT_BY

                    # 디버깅을 위해 추가 (터미널/콘솔에 출력됨)
                    print(f"DEBUG (report page): all_recommendations_cache 생성됨, 개수: {len(st.session_state.all_recommendations_cache)}")
                    print(f"DEBUG (report page): num_items_currently_showing: {st.session_state.num_items_currently_showing}")

                    navigate_to("recommend")
                except Exception as e:
                    st.error(f"추천 식품을 생성하는 중 오류가 발생했습니다: {e}")
                    print(f"ERROR (report page) during recommend_foods call: {e}") # 터미널/콘솔에 오류 출력
                    # 오류 발생 시 세션 상태 초기화
            navigate_to("recommend")

# ✅ 4단계: 식품 추천
elif st.session_state.page == 'recommend':
    st.title("🥗 식품 추천 결과")
    st.markdown("회원님의 건강 분석 결과와 BMR을 바탕으로 맞춤형 식품을 추천해 드립니다. 건강 점수가 높을수록 현재 상태에 더 적합한 식품입니다.")

    if 'all_recommendations_cache' not in st.session_state or \
       'num_items_currently_showing' not in st.session_state or \
       'lowest_foods_cache' not in st.session_state:
        st.warning("추천 식품 정보를 불러오는 중 오류가 발생했습니다. 이전 페이지로 돌아가 다시 시도해주세요.")
        if st.button("이전 페이지로 돌아가기"):
            go_back()
    else:
        all_cached_foods = st.session_state.all_recommendations_cache
        lowest_foods_to_display = st.session_state.lowest_foods_cache

        # --- 필터링 UI 및 로직 (기존과 동일) ---
        with st.expander("🔍 필터 옵션", expanded=False):
            filter_col1, filter_col2 = st.columns(2)

            with filter_col1:
                MAIN_CATEGORY_FIELD_NAME = '식품분류'
                raw_main_categories = [food.get(MAIN_CATEGORY_FIELD_NAME) for food in all_cached_foods if food]
                valid_main_categories = [str(cat) for cat in raw_main_categories if cat is not None and str(cat).strip() != "" and str(cat) not in ['0', 0]]
                unique_main_categories = sorted(list(set(valid_main_categories)))

                if not unique_main_categories:
                    unique_main_categories = ['사용 가능한 분류 없음']

                if 'selected_main_categories_filter' not in st.session_state:
                    st.session_state.selected_main_categories_filter = []

                current_selection_for_main_categories = st.multiselect(
                    "식품 대분류 선택 (다중 선택 가능):",
                    options=unique_main_categories,
                    default=st.session_state.selected_main_categories_filter,
                    key="filter_main_categories_multiselect_key"
                )
                st.session_state.selected_main_categories_filter = current_selection_for_main_categories

            with filter_col2:
                if 'max_sodium_filter' not in st.session_state:
                    st.session_state.max_sodium_filter = None

                max_sodium_input_val = st.number_input(
                    "최대 나트륨 (mg 이하):",
                    min_value=0,
                    value=st.session_state.max_sodium_filter,
                    step=50,
                    placeholder="숫자 입력 (예: 500)",
                    key="filter_max_sodium_input_val_key"
                )
                st.session_state.max_sodium_filter = max_sodium_input_val

            if st.button("필터 초기화", key="reset_all_filters_button"):
                st.session_state.selected_main_categories_filter = []
                st.session_state.max_sodium_filter = None
                st.rerun()

        st.markdown("---")

        # 필터 적용 로직
        filtered_foods = all_cached_foods

        if st.session_state.selected_main_categories_filter:
            filtered_foods = [
                food for food in filtered_foods
                if food and food.get(MAIN_CATEGORY_FIELD_NAME) in st.session_state.selected_main_categories_filter
            ]

        if st.session_state.max_sodium_filter is not None:
            filtered_foods = [
                food for food in filtered_foods
                if food and food.get('나트륨(mg)', float('inf')) <= st.session_state.max_sodium_filter
            ]

        # --- 상위 추천 식품 목록 섹션 (st.expander로 감싸기) ---
        # 여기를 변경하여 버튼을 누르면 보이도록 합니다.
        with st.expander("✨ 추천 식품 목록 보기", expanded=True): # 기본적으로 펼쳐져 있게 하거나 (True), 닫혀있게 (False) 선택
            num_to_display_now = st.session_state.num_items_currently_showing
            recommendations_to_render = filtered_foods[:num_to_display_now]

            if not recommendations_to_render:
                st.info("선택하신 필터 조건에 맞는 추천 식품이 없거나, 표시할 추천이 더 이상 없습니다.")
            else:
                subheader_text = "" # Expander 제목이 있으므로 서브헤더는 제거하거나 간소화
                if st.session_state.selected_main_categories_filter or \
                   st.session_state.max_sodium_filter is not None:
                    subheader_text = " (필터 적용됨)"
                st.markdown(f"**건강 점수가 높은 추천 식품들입니다.{subheader_text}**") # Expander 내부 텍스트

                for food_item in recommendations_to_render:
                    with st.container(border=True):
                        col1, col_detail = st.columns([0.6, 0.4])
                        with col1:
                            st.subheader(food_item['식품명'])
                            score_badge_html = get_styled_score_display(food_item['health_score'])
                            st.markdown(f"**⭐ 건강 점수:** {score_badge_html}", unsafe_allow_html=True)
                            penalties = food_item.get('penalties', {})
                            if any(p > 0 for p in penalties.values()):
                                st.markdown("**주요 감점 요인:**")
                                sorted_penalties = sorted([(k, v) for k, v in penalties.items() if v > 0], key=lambda x: x[1], reverse=True)
                                for nutrient, penalty_value in sorted_penalties[:2]:
                                    st.caption(f"- {nutrient}: 감점 {penalty_value:.2f}점")
                            else:
                                st.caption("특별한 감점 요인이 없는 우수한 식품입니다!")

                        with col_detail:
                            st.write("")
                            st.write("")
                            if st.button("상세 정보 보기", key=f"detail_{food_item['식품명']}"):
                                st.session_state.selected_food_details = food_item
                                navigate_to("food_detail")

                if num_to_display_now < len(filtered_foods):
                    st.markdown("---")
                    if st.button("더 많은 추천 보기 ➕", use_container_width=True, type="secondary"):
                        st.session_state.num_items_currently_showing += st.session_state.get('increment_amount', 5)
                        if st.session_state.num_items_currently_showing > len(filtered_foods):
                            st.session_state.num_items_currently_showing = len(filtered_foods)
                        st.rerun()
                elif len(filtered_foods) > 0 and len(filtered_foods) >= st.session_state.get('INITIAL_ITEMS_TO_SHOW', 5) :
                    st.success("✅ 모든 추천 식품을 확인했습니다.")

        st.markdown("---")

        # --- 최하위 건강 점수 식품 섹션 (st.expander로 감싸기) ---
        # 여기도 버튼을 누르면 보이도록 변경합니다.
        with st.expander("⚠️ 건강 점수가 낮은 식품 보기", expanded=False): # 기본적으로 닫혀있음
            st.info("아래 목록은 건강 점수가 낮은 편에 속하는 식품들입니다. 섭취 시 영양 성분을 확인하거나, 섭취량을 조절하는 것을 권장합니다.")

            if lowest_foods_to_display:
                for i, food in enumerate(lowest_foods_to_display):
                    with st.container(border=True):
                        col_low1, col_low2 = st.columns([0.6, 0.4])
                        with col_low1:
                            st.markdown(f"**{i+1}. {food['식품명']}**")
                            score_badge_html = get_styled_score_display(food['health_score'])
                            st.markdown(f"**📉 건강 점수:** {score_badge_html}", unsafe_allow_html=True)

                            penalties = food.get('penalties', {})
                            if any(p > 0 for p in penalties.values()):
                                st.markdown("**주요 감점 요인:**")
                                sorted_penalties = sorted([(k, v) for k, v in penalties.items() if v > 0], key=lambda x: x[1], reverse=True)
                                for nutrient, penalty_value in sorted_penalties[:2]:
                                    st.caption(f"- {nutrient}: 감점 {penalty_value:.2f}점")
                            else:
                                st.caption("특별한 감점 요인이 없습니다. (전반적인 점수가 낮은 경우)")
                        with col_low2:
                            st.write("")
                            st.write("")
                            if st.button("상세 정보 보기", key=f"low_detail_{food['식품명']}"):
                                st.session_state.selected_food_details = food
                                navigate_to("food_detail")
            else:
                st.info("최하위 건강 점수 식품 목록을 불러올 수 없습니다.")

        st.markdown("---")


        if st.button("◀ 이전 (건강 분석 결과)"):
            keys_to_delete = ['selected_main_categories_filter', 'max_sodium_filter',
                              'all_recommendations_cache', 'num_items_currently_showing', 'increment_amount',
                              'lowest_foods_cache']
            for key in st.session_state.keys():
                if key in keys_to_delete:
                    del st.session_state[key]
            go_back()

# ✅ 5단계: 식품 상세 Nutrient View
elif st.session_state.page == 'food_detail':
    # 'selected_food_details'가 세션 상태에 있는지 확인
    if 'selected_food_details' not in st.session_state:
        st.error("상세 정보를 불러올 식품이 선택되지 않았습니다. 추천 목록으로 돌아가 다시 선택해주세요.")
        if st.button("◀ 추천 목록으로 돌아가기"):
            go_back()
        st.stop() # 스크립트 실행 중지

    food_item = st.session_state.selected_food_details # 수정된 부분
    selected_food_name = food_item['식품명'] # 수정된 부분

    st.markdown(f"<h1 style='text-align:center;'>{selected_food_name}</h1>", unsafe_allow_html=True) # 중앙 정렬 제목
    score_badge_html = get_styled_score_display(food_item['health_score']) # 수정된 부분
    st.markdown(f"<h3 style='text-align:center;'>⭐ 건강 점수: {score_badge_html}</h3>", unsafe_allow_html=True)
    st.divider()

    # 원본 food_df에서 해당 식품의 다른 모든 컬럼을 가져와야 한다면 (예: '식품중량', '에너지(kcal)' 등)
    # 아래와 같이 다시 찾아오는 것이 더 안전합니다.
    food_row = food_df[food_df['식품명'] == selected_food_name].iloc[0]


    # --- DASH 다이어트 기준 불러오기 ---
    dash_targets = {} # 기본값으로 빈 딕셔너리 초기화
    if 'bmr' in st.session_state and st.session_state.bmr is not None and st.session_state.bmr > 0:
        try:
            dash_targets = adjust_dash_standard(st.session_state.bmr)
        except Exception as e:
            st.error(f"DASH 기준을 불러오는 중 오류 발생: {e}")
            dash_targets = {} # 오류 시 빈 딕셔너리로 처리
    else:
        st.info("사용자 BMR 정보가 없어 개인별 DASH 기준 비교는 제공되지 않습니다. 일반 영양 정보만 표시됩니다.")
        # 이 경우 dash_targets는 위에서 초기화된 {} 그대로 사용됨.

    # --- 기본 정보 섹션 ---
    st.subheader("📋 기본 정보")
    col_info1, col_info2, col_info3 = st.columns(3) # 3열로 확장
    with col_info1:
        serving_size_g = food_row.get('식품중량', food_row.get('영양성분함량기준량'))
        display_serving_size = f"{serving_size_g}" if isinstance(serving_size_g, (int, float,str)) else "정보없음"
        st.metric("제공량 기준", display_serving_size)

    with col_info2:
        energy_kcal = food_row.get('에너지(kcal)')
        display_energy = f"{energy_kcal} kcal" if isinstance(energy_kcal, (int, float,np.number)) else "정보없음"
        st.metric("에너지", display_energy)
    with col_info3: # 새로운 컬럼 추가 (예: 분류)
        st.metric("식품 분류", food_row.get('식품대분류명', '미분류'))

    st.caption(f"제조사: {food_row.get('제조사명', '정보 없음')}")
    st.divider()

    # --- 매크로 영양소 분석 (파이 차트) ---
    st.subheader("📊 주요 영양소 구성 (칼로리 기반 추정)")
    macros = {'탄수화물(g)': 4, '단백질(g)': 4, '지방(g)': 9} # g당 kcal
    macro_values_kcal = {}
    total_macros_kcal = 0
    for nutrient, kcal_per_g in macros.items():
        value_g = food_row.get(nutrient)
        if isinstance(value_g, (int, float)) and value_g > 0:
            kcal_value = value_g * kcal_per_g
            macro_values_kcal[nutrient.split('(')[0]] = kcal_value # 이름에서 단위 제거
            total_macros_kcal += kcal_value

    if total_macros_kcal > 0 : # 총 매크로 칼로리가 0보다 클 때만 차트 표시
        macro_df = pd.DataFrame(list(macro_values_kcal.items()), columns=['영양소', '칼로리 기여도(kcal)'])
        if not macro_df.empty:
            fig_pie = px.pie(macro_df, values='칼로리 기여도(kcal)', names='영양소',
                             hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label',
                                  hovertemplate="<b>%{label}</b><br>칼로리: %{value:.1f}kcal (%{percent})") # 호버 템플릿 추가
            fig_pie.update_layout(height=350) # 차트 높이 조절
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown(f"<p style='text-align:center; font-weight:bold;'>총 매크로 칼로리 기여: {total_macros_kcal:.1f} kcal</p>", unsafe_allow_html=True)
        else:
            st.caption("주요 영양소(탄수화물, 단백질, 지방) 정보가 부족하여 차트를 표시할 수 없습니다.")

    else:
        st.caption("주요 영양소(탄수화물, 단백질, 지방) 정보가 부족하여 칼로리 구성 비율을 계산할 수 없습니다.")
    st.divider()


    # --- 혈압 관리 핵심 성분 (DASH 기준 비교 및 바 차트) ---
    st.subheader("⚙️ 혈압 관리 핵심 성분 (DASH 기준 비교)")
    st.info("💡 아래 영양소들은 혈압 관리에 특히 중요한 성분들입니다. DASH 다이어트 개인 목표치와 비교하여 확인하세요.")
    key_nutrients_for_bp_dash = [ # DASH 기준과 비교할 주요 영양소 목록
        '나트륨(mg)', '칼륨(mg)', '식이섬유(g)',
        '포화지방산(g)', '콜레스테롤(mg)', '칼슘(mg)'
        # '마그네슘(mg)' # 데이터에 있다면 추가
    ]

    # 각 영양소에 대한 반복 로직 (2열로 배치)
    num_cols = 2
    cols = st.columns(num_cols)
    for i, nutrient in enumerate(key_nutrients_for_bp_dash):
        with cols[i % num_cols]:
            value = food_row.get(nutrient)
            if not isinstance(value, (int, float)): # 숫자형이 아니면 스킵
                st.markdown(f"**{nutrient.split('(')[0]}**: 정보 없음")
                continue

            unit = nutrient[nutrient.find("(")+1:nutrient.find(")")]
            target_value = None
            comparison_text = ""
            current_value_display = f"{value:.1f} {unit}"
            delta_color = "off" # Streamlit delta_color 기본값
            delta_text = ""

            if dash_targets and nutrient in dash_targets and dash_targets.get(nutrient,0) > 0: # dash_targets에 값이 있고 0보다 클때
                target_value = dash_targets[nutrient]
                percentage_of_dash = (value / target_value) * 100 if target_value > 0 else 0

                if nutrient in ['나트륨(mg)', '포화지방산(g)', '콜레스테롤(mg)']: # 낮을수록 좋은 경우
                    comparison_text = f"목표: {target_value:.1f}{unit} 이하"
                    if value > target_value: # 목표치 초과
                        delta_text = f"높음 ({percentage_of_dash:.0f}%)"
                        delta_color = "inverse" # Streamlit의 역색상 (빨간색)
                    else: # 목표치 이하 또는 적정
                        delta_text = f"양호 ({percentage_of_dash:.0f}%)"
                        delta_color = "normal" # Streamlit의 기본색상 (녹색)
                else: # 높을수록 좋은 경우 (칼륨, 식이섬유, 칼슘 등)
                    comparison_text = f"목표: 약 {target_value:.1f}{unit}"
                    if value >= target_value * 0.9: # 90% 이상 달성
                        delta_text = f"충분 ({percentage_of_dash:.0f}%)"
                        delta_color = "normal"
                    elif value >= target_value * 0.5: # 50% 이상 달성
                        delta_text = f"보통 ({percentage_of_dash:.0f}%)"
                        delta_color = "off" # 회색 (큰 변화 없음)
                    else: # 50% 미만 달성
                        delta_text = f"부족 ({percentage_of_dash:.0f}%)"
                        delta_color = "inverse" # (빨간색)

            st.metric(label=f"{nutrient.split('(')[0]}", value=current_value_display, delta=delta_text, delta_color=delta_color)

            # Popover는 아이콘 옆에 작게 넣거나, 별도 섹션으로
            with st.popover(f"설명"): # key 인자 사용 (Streamlit 1.34.0 이상 필요)
                if nutrient in NUTRIENT_INFO_POPOVER:
                    st.markdown(NUTRIENT_INFO_POPOVER[nutrient])
                    if comparison_text:
                        st.caption(f"나의 DASH 목표: {comparison_text}")
                    elif nutrient in DAILY_VALUES and DAILY_VALUES[nutrient] > 0:
                         st.caption(f"일반 1일 기준치: {DAILY_VALUES[nutrient]} {unit}")
                else:
                    st.caption("해당 영양소에 대한 추가 정보가 없습니다.")
            st.markdown("---") # 각 핵심 영양소 항목 사이에 구분선
    st.divider()

    # --- 기타 영양 정보 섹션 (기존 방식 유지 또는 유사하게 개선 가능) ---
    st.subheader("🍽️ 기타 영양 정보")
    other_display_nutrients = ["단백질(g)", "탄수화물(g)", "지방(g)", "당류(g)"] # 표시할 기타 영양소

    nutrient_data = []
    for nutrient in other_display_nutrients:
        if nutrient in food_row:
            value = food_row.get(nutrient)
            if isinstance(value, (int,float)):
                unit = nutrient[nutrient.find("(")+1:nutrient.find(")")]
                display_value = f"{value:.1f}{unit}"
                dv_percentage = ""
                if nutrient in DAILY_VALUES and DAILY_VALUES[nutrient] > 0:
                    dv_percentage_val = (value / DAILY_VALUES[nutrient]) * 100
                    dv_percentage = f"{dv_percentage_val:.0f}%"
                nutrient_data.append([nutrient.split('(')[0], display_value, dv_percentage])
            else:
                nutrient_data.append([nutrient.split('(')[0], "정보없음", ""])

    nutrient_df = pd.DataFrame(nutrient_data, columns=["영양소", "함량", "1일 기준치(DV)"])
    st.dataframe(nutrient_df, hide_index=True)
    st.caption("DV: Daily Value (일반 1일 영양성분 기준치)")
    st.divider()

    if st.button("◀ 추천 목록으로 돌아가기", use_container_width=True):
        go_back()