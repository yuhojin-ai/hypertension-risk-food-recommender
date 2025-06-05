import streamlit as st
import requests
import base64
import json
import re
from PIL import Image
import pandas as pd
import joblib

# ✅ CLOVA OCR 설정
CLOVA_OCR_URL = "https://gwftjuxmag.apigw.ntruss.com/custom/v1/42351/75174c8cae3b575cc5cb61145c0898ed94fe524a3acb4ef70537b5fe6fcdccac/general"
CLOVA_OCR_SECRET = "THNtdWh3RlZFQmdyT2VjU0RPT2tOVkxDUVpvYnFwZ04="

# ✅ 초기 구성
st.set_page_config(page_title="CareBite OCR", layout="centered")

# ✅ 스타일
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        h1, h2, h3 { color: #004080; }
    </style>
""", unsafe_allow_html=True)

st.title(" HiNavi OCR: 건강검진 기반 고혈압 예측")
st.caption("주요 수치를 자동 추출하고 예측 모델에 활용")

# ✅ OCR 요청
def request_clova_ocr(image_bytes, filename="image.jpg"):
    headers = {"X-OCR-SECRET": CLOVA_OCR_SECRET, "Content-Type": "application/json"}
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    payload = {
        "version": "V2", "requestId": "sample_id", "timestamp": 0,
        "images": [{"name": filename, "format": "jpg", "data": encoded_image}]
    }
    response = requests.post(CLOVA_OCR_URL, headers=headers, data=json.dumps(payload))
    return response.json()

# ✅ 수치 추출 + 보정

def extract_number(keywords, text):
    for keyword in keywords:
        try:
            pattern = rf"{keyword}[^\n\d]*(\d+\.?\d*)"
            match = re.search(pattern, text)
            if match:
                return float(match.group(1))
        except: pass
    return None

def clean_feature_value(key, value):
    normal_ranges = {
        '혈청크레아티닌': (0.5, 2.0), '혈색소': (8, 20), '식전혈당(공복혈당)': (50, 300),
        '감마지티피': (10, 500), '트리글리세라이드': (30, 500), 'HDL콜레스테롤': (10, 100)
    }
    if value is None or key not in normal_ranges: return value
    min_v, max_v = normal_ranges[key]
    if value > max_v and value / 10 < max_v:
        return round(value / 10, 1)
    return value

# 혈압 기반 status 생성

def extract_blood_pressure_pair(text):
    match = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def extract_hypertension_status(text):
    sbp, dbp = extract_blood_pressure_pair(text)
    if sbp and dbp:
        if sbp >= 140 or dbp >= 90: return 1
        elif 120 <= sbp <= 139 or 80 <= dbp <= 89: return 0
    return 0

# ✅ 연령대 코드 계산

def calculate_age_group(age):
    return (age // 5)

# ✅ 키/체중 한 줄에서 추출

def extract_height_weight_pair(text):
    # 예: "키(cm) 및 몸무게(kg) 150 / 70"
    match = re.search(r"키[^\d]*(\d{2,3})[^\d]{0,5}[/][^\d]{0,5}(\d{2,3})", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

# ✅ 모델 로딩
@st.cache_resource
def load_model():
    return joblib.load("notebooks/logistic_model.pkl")

model = load_model()
model_features = [
    '연령대코드(5세단위)', '식전혈당(공복혈당)', '혈색소', '혈청크레아티닌',
    '감마지티피', 'bmi', '트리글리세라이드', 'HDL콜레스테롤'
]

# ✅ 사용자 입력
st.header("1️ 사용자 정보 입력")
with st.form("user_info_form"):
    gender = st.selectbox("성별", ["남", "여"])
    age = st.number_input("나이", min_value=10, max_value=100, step=1)
    image = st.file_uploader("건강검진표 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
    submitted = st.form_submit_button("다음 단계로 ➡️")

# ✅ 분석 실행
if submitted:
    if not image:
        st.warning(" 이미지를 업로드해주세요.")
    else:
        st.success(" 사용자 정보 입력 완료!")
        st.image(image, caption="업로드된 검진표", use_container_width=True)

        with st.spinner(" OCR 분석 중..."):
            image_bytes = image.read()
            result = request_clova_ocr(image_bytes, image.name)
            fields = result.get("images", [])[0].get("fields", [])
            raw_text = " ".join([f.get("inferText", "") for f in fields])

        st.markdown("### OCR 전체 추출 텍스트")
        st.text_area("텍스트", raw_text, height=200)

        # 키 체중 추출
        height_cm, weight_kg = extract_height_weight_pair(raw_text)
        if height_cm is None:
            height_cm = extract_number(["키"], raw_text)
        if weight_kg is None:
            weight_kg = extract_number(["몸무게"], raw_text)

        bmi = round(weight_kg / ((height_cm / 100) ** 2), 2) if height_cm and weight_kg else 0

        features = {
            '연령대코드(5세단위)': calculate_age_group(age),
            '식전혈당(공복혈당)': clean_feature_value('식전혈당(공복혈당)', extract_number(["공복혈당"], raw_text)),
            '혈색소': clean_feature_value('혈색소', extract_number(["혈색소"], raw_text)),
            '혈청크레아티닌': clean_feature_value('혈청크레아티닌', extract_number(["크레아티닌", "크레아티"], raw_text)),
            '감마지티피': clean_feature_value('감마지티피', extract_number(["감마지티피", "GTP"], raw_text)),
            '트리글리세라이드': clean_feature_value('트리글리세라이드', extract_number(["중성지방"], raw_text)),
            'HDL콜레스테롤': clean_feature_value('HDL콜레스테롤', extract_number(["고밀도 콜레스테롤"], raw_text)),
            'bmi': bmi,
            'hypertension_status': extract_hypertension_status(raw_text)
        }
        st.markdown("#### 📐 OCR 추출된 키/체중")
        st.write(f"- 키(cm): **{height_cm}**")
        st.write(f"- 몸무개개(kg): **{weight_kg}**")
        st.write(f"- 계산된 BMI: **{bmi}**")

        st.markdown("###  최종 모델 입력 피처")
        st.json(features)

        st.session_state["user_features"] = features

        st.markdown("---")
        st.header(" 고혈압 예측 결과")
        input_df = pd.DataFrame([{k: features.get(k, 0) for k in model_features}])
        proba = model.predict_proba(input_df)[0][1]
        st.subheader(f"예측된 고혈압 확률: **{proba * 100:.2f}%**")

        if proba >= 0.7:
            st.error(" 고위험군입니다. 생활습관 개선이 필요합니다.")
        elif proba >= 0.4:
            st.warning("주의 단계입니다. 혈압 관리가 필요합니다.")
        else:
            st.success("정상 범위로 예측되었습니다.")