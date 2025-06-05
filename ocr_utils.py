# ocr_utils.py

import re
import requests
import json
import base64 # base64 모듈 임포트 추가
import streamlit as st # st.secrets를 사용하기 위해 임포트
import time # time 모듈 임포트 추가 (requestId, timestamp 생성용)

# --- CLOVA OCR API 정보 로드 ---
@st.cache_resource # API 정보 로딩을 캐싱
def get_clova_ocr_info():
    """st.secrets에서 CLOVA OCR API 정보를 로드합니다."""
    try:
        api_info = st.secrets["clova_ocr"]
        return {
            "api_gateway_key": api_info["api_gateway_key"],
            "api_url": api_info["api_url"]
        }
    except KeyError as e:
        # 특정 키가 없는 경우 명확한 오류 메시지
        st.error(f"CLOVA OCR API secrets에 필요한 키({e})가 Streamlit secrets에 설정되어 있지 않습니다. "
                 f".streamlit/secrets.toml 파일을 확인해주세요.")
        return None
    except Exception as e:
        st.error(f"CLOVA OCR 정보 로드 중 오류 발생: {e}")
        return None

def extract_number(keywords, text):
    for keyword in keywords:
        pattern = rf"{keyword}[^\d\n]*?(\d+\.?\d*)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None

def extract_height_weight_pair(text):
    # 1. '숫자 / 숫자 키(cm) 및 몸무게(kg)' 역순 패턴 (새로 추가)
    # 예: '170 / 80 키(cm) 및 몸무게(kg)'
    # 숫자를 먼저 찾고, 그 뒤에 키/몸무게 관련 키워드가 따라오는 패턴
    match = re.search(r"(\d{2,3}(?:\.\d)?)\s*[/]\s*(\d{2,3}(?:\.\d)?)\s*(?:키|신장)\s*\(cm\)\s*(?:및\s*)?(?:몸무게|체중)\s*\(kg\)", text, re.IGNORECASE)
    if match:
        return float(match.group(1)), float(match.group(2))

    # 2. '키(cm) 및 몸무게(kg) 숫자 / 숫자' 패턴 (기존, 순서 보정)
    # 예: '키(cm) 및 몸무게(kg) 170 / 80' 또는 '키(cm) 및 몸무게(kg) 170 / 80'
    match = re.search(r"(?:키|신장)\s*\(cm\)\s*(?:및\s*)?(?:몸무게|체중)\s*\(kg\)\s*(\d{2,3}(?:\.\d)?)\s*[/]\s*(\d{2,3}(?:\.\d)?)", text, re.IGNORECASE)
    if match:
        return float(match.group(1)), float(match.group(2))

    # 3. '키 숫자 / 숫자' 패턴 (기존 유지, 더 일반적)
    # 예: 키 170 / 65
    match = re.search(r"(?:키|신장)[^\d]*(\d{2,3}(?:\.\d)?)[^\d]{0,5}[/][^\d]{0,5}(\d{2,3}(?:\.\d)?)", text, re.IGNORECASE)
    if match:
        return float(match.group(1)), float(match.group(2))

    # 4. '키 숫자 몸무게 숫자' 패턴 (기존 유지)
    # 예: 키 170 몸무게 65
    match = re.search(r"(?:키|신장)\s*(\d{2,3}(?:\.\d)?)\s*(?:cm|CM)?\s*(?:몸무게|체중)\s*(\d{2,3}(?:\.\d)?)", text, re.IGNORECASE)
    if match:
        return float(match.group(1)), float(match.group(2))

    # 5. 다른 일반적인 키/몸무게 패턴 (개별적으로 찾기)
    height_match = re.search(r"(?:키|신장)[^\d\n]*?(\d{2,3}(?:\.\d)?)\s*(?:cm|CM)?", text, re.IGNORECASE)
    weight_match = re.search(r"(?:몸무게|체중)[^\d\n]*?(\d{2,3}(?:\.\d)?)\s*(?:kg|KG)?", text, re.IGNORECASE)

    h = float(height_match.group(1)) if height_match else None
    w = float(weight_match.group(1)) if weight_match else None

    if h is not None and w is not None:
        return h, w

    return None, None


def clean_feature_value(key, value):
    if value is None or not isinstance(value, (int, float)):
        return value

    ranges = {
        '혈청크레아티닌': (0.5, 2.0),
        '혈색소': (8, 20),
        '식전혈당(공복혈당)': (50, 300),
        '감마지티피': (10, 500),
        '트리글리세라이드': (30, 500),
        'HDL콜레스테롤': (10, 100)
    }
    if key not in ranges:
        return value
    min_v, max_v = ranges[key]

    if value > max_v * 10 and value / 10 >= min_v and value / 10 <= max_v :
        return round(value / 10, 1)
    elif value > max_v and value / 100 >= min_v and value / 100 <= max_v :
        return round(value / 100, 1)
    return value

def calculate_age_group(age):
    return age // 5

# --- 사용자가 제시한 request_clova_ocr 함수 ---
def request_clova_ocr(image_bytes, filename="image.jpg", clova_info=None):
    if clova_info is None:
        raise ValueError("CLOVA OCR API 정보가 제공되지 않았습니다.")

    headers = {
        "X-OCR-SECRET": clova_info["api_gateway_key"],
        "Content-Type": "application/json"
    }
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')

    # 누락된 requestId, timestamp를 동적으로 생성
    current_time_ms = int(time.time() * 1000)
    payload = {
        "version": "V1", # API 문서에 따라 V1 또는 V2 선택 (일반적으로 V1이면 충분)
        "requestId": f"streamlit_ocr_{current_time_ms}", # 고유한 요청 ID
        "timestamp": current_time_ms, # 현재 시간 (밀리초)
        "images": [{"name": filename, "format": "jpg", "data": encoded_image}]
    }

    response = requests.post(clova_info["api_url"], headers=headers, data=json.dumps(payload))
    response.raise_for_status() # HTTP 오류 발생 시 예외 발생

    return response.json()

@st.cache_data(show_spinner="CLOVA OCR 텍스트를 추출 중입니다...")
def detect_text_from_image(uploaded_file):
    """
    CLOVA OCR API를 사용하여 이미지에서 텍스트를 추출합니다.
    uploaded_file: Streamlit의 uploaded_file 객체
    """
    clova_info = get_clova_ocr_info()
    if clova_info is None:
        # get_clova_ocr_info에서 이미 st.error를 출력했으므로 여기서는 추가 메시지 없이 반환
        return "CLOVA OCR 클라이언트를 초기화할 수 없어 텍스트를 추출할 수 없습니다."

    try:
        image_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name

        # 수정된 request_clova_ocr 함수 호출
        ocr_response_json = request_clova_ocr(image_bytes, filename=filename, clova_info=clova_info)

        # CLOVA OCR 응답 파싱
        full_text = ""
        # 응답 형식에 따라 파싱 로직을 조정해야 합니다.
        # 일반적인 CLOVA OCR 응답은 'images' 배열 내에 'result' 또는 'fields'를 가집니다.
        # 특히 document/validation 또는 general API의 경우, 'result.text'에 전체 텍스트가 있을 수 있습니다.

        if ocr_response_json and ocr_response_json.get("images"):
            for image_result in ocr_response_json["images"]:
                if image_result.get("result") and image_result["result"].get("text"):
                    full_text = image_result["result"]["text"]
                    break # 첫 번째 이미지의 전체 텍스트만 가져오는 것으로 가정
                elif image_result.get("fields"):
                    for field in image_result["fields"]:
                        full_text += field.get("inferText", "") + " "
            if not full_text: # fields에서도 텍스트를 찾지 못했을 경우
                st.warning(f"CLOVA OCR 응답에서 'fields'나 'result.text'를 찾을 수 없습니다. 응답 구조 확인 필요: {ocr_response_json}")
                return "CLOVA OCR 응답 구조가 예상과 다릅니다. 콘솔에서 응답을 확인해주세요."
        elif ocr_response_json and ocr_response_json.get("text"): # 다른 형태의 전체 텍스트 필드 (legacy 또는 다른 API)
            full_text = ocr_response_json["text"]
        else:
            st.warning(f"CLOVA OCR 응답에서 텍스트를 찾을 수 없습니다. 응답: {ocr_response_json}")
            return "CLOVA OCR 응답에서 유효한 텍스트를 찾을 수 없습니다."

        return full_text.strip()

    except requests.exceptions.HTTPError as http_err:
        return f"HTTP 오류 발생: {http_err} - 응답: {http_err.response.text}"
    except requests.exceptions.ConnectionError as conn_err:
        return f"연결 오류 발생: {conn_err}"
    except requests.exceptions.Timeout as timeout_err:
        return f"요청 시간 초과: {timeout_err}"
    except requests.exceptions.RequestException as req_err:
        return f"요청 오류 발생: {req_err}"
    except json.JSONDecodeError:
        return f"CLOVA OCR 응답 파싱 오류: 유효한 JSON이 아닙니다. 응답: {response.text}"
    except Exception as e:
        return f"알 수 없는 OCR 처리 오류 발생: {e}"