import numpy as np

# NORMALIZATION_RULES = {
#     '혈색소':                 {'min': 11,  'risk_max': 18,  'direction': 'low'},
#     '트리글리세라이드':         {'min': 50,  'risk_max': 200, 'direction': 'high'},
#     'HDL콜레스테롤':           {'min': 45,  'risk_max': 100, 'direction': 'low'},
#     '감마지티피':              {'min': 10,  'risk_max': 70,  'direction': 'high'},
#     '식전혈당(공복혈당)':       {'min': 70,  'risk_max': 125, 'direction': 'high'},
# }

NORMALIZATION_RULES = {
    '혈색소':                 {'min': 11,  'risk_max': 18,  'direction': 'low'},
    '트리글리세라이드':         {'min': 50,  'risk_max': 200, 'direction': 'high'},
    'HDL콜레스테롤':           {'min': 45,  'risk_max': 100, 'direction': 'low'},
    '감마지티피':              {'min': 10,  'risk_max': 70,  'direction': 'high'},
    '식전혈당(공복혈당)':       {'min': 70,  'risk_max': 125, 'direction': 'high'},
    '연령대코드(5세단위)':      {'min': 4,  'risk_max': 16,  'direction': 'high'},
    'bmi':                    {'min': 18,  'risk_max': 23, 'direction': 'high'}
}

COEFFICIENTS = {
    '혈색소': 0.11,
    '트리글리세라이드': 0.09,
    'HDL콜레스테롤': 0.08,
    '감마지티피': 0.06,
    '식전혈당(공복혈당)': 0.05,
    '연령대코드(5세단위)': 0.43,
    'bmi': 0.18
}

# COEFFICIENTS = {
#     '혈색소': 0.2639,
#     '트리글리세라이드': 0.2361,
#     'HDL콜레스테롤': 0.1944,
#     '감마지티피': 0.1667,
#     '식전혈당(공복혈당)': 0.1389,
# }

def normalize_health_inputs(user_values, normalization_rules):
    """
    사용자 전체 건강 수치를 일괄 정규화하는 함수
    user_values: {'항목명': 수치, ...}
    normalization_rules: {'항목명': {'min': x, 'risk_max': y, 'direction': z}, ...}
    
    return: {'항목명': 정규화 점수}
    """
    normalized_scores = {}

    for key, value in user_values.items():
        if key in normalization_rules:
            rule = normalization_rules[key]
            min_val = rule['min']
            risk_max = rule['risk_max']
            direction = rule['direction']
            
            try:
                norm = (value - min_val) / (risk_max - min_val)
                if direction == 'low':
                    norm = 1 - norm
                norm = max(0, min(1, norm))
                normalized_scores[key] = round(norm, 4)
            except ZeroDivisionError:
                normalized_scores[key] = 0.0  # 예외 처리
        else:
            normalized_scores[key] = None  # 기준이 없는 경우

    return normalized_scores

def calculate_risk_score(normalized_scores, coefficients):
    """
    정규화된 수치와 회귀 계수를 곱해 최종 위험도 점수 계산
    """
    score = 0.0
    for key, norm_value in normalized_scores.items():
        coef = coefficients.get(key, 0)
        if norm_value is not None:
            score += norm_value * coef
    return round(score, 4)

def classify_risk(score):
    if score <= 0.4:
        return '양호'
    elif score <= 0.65:
        return '주의'
    elif score <= 0.85:
        return '위험'
    else:
        return '고위험'
    
    
def calculate_bmr_mifflin(sex: str, age: int, height_cm: float, weight_kg: float) -> float:
    """
    BMR 계산 (Mifflin–St Jeor 공식 기준, 최신 표준)
    - sex: '남성' 또는 '여성'
    - age: 나이 (년)
    - height_cm: 키 (cm)
    - weight_kg: 몸무게 (kg)
    - 반환값: 하루 기초대사량 (kcal), 소수점 1자리
    """
    sex = sex.lower()
    
    if sex == '남성':
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    elif sex == '여성':
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    else:
        raise ValueError("성별은 '남성' 또는 '여성'이어야 합니다.")
    
    return round(bmr, 1)

def compute_metabolic_score(bmr: float, bmi: float) -> float:
    """
    BMR과 BMI를 기반으로 0~1 사이의 metabolic_score를 계산.
    값이 1에 가까울수록 대사 건강 상태가 좋음을 의미함.
    
    - BMR 정규화 기준: 1200~2500 kcal
    - BMI 정규화 기준: 중심 22, ±10 범위
    - 가중치: BMR(40%), BMI(60%)
    """
    # BMR 정규화 (1200~2500 kcal 기준)
    normalized_bmr = (bmr - 1200) / (2500 - 1200)
    normalized_bmr = max(0, min(1, normalized_bmr))  # 안정성 확보
    
    # BMI 점수화 (22 중심, 거리 기반 감점)
    bmi_distance = abs(bmi - 22)
    normalized_bmi_score = max(0, 1 - (bmi_distance / 10))

    # 가중 평균 계산
    metabolic_score = (normalized_bmr * 0.4) + (normalized_bmi_score * 0.6)
    
    return round(metabolic_score, 3)

def compute_health_index(risk_score: float, metabolic_score: float, a: float = 0.7) -> float:
    """
    risk_score와 metabolic_score를 기반으로 health_index 계산
    - risk_score: 0~1 범위 (예측된 고혈압 위험도)
    - metabolic_score: 0~1 범위
    - a: 위험도에 부여할 가중치 (기본값 0.7)
    - 반환값: 0~1 범위의 health_index (높을수록 성분에 민감)
    """
    
    health_index = (risk_score * a) + ((1 - metabolic_score) * (1 - a))
    return round(health_index, 3)

b_values = {
    '나트륨(mg)': 0.5,
    '포화지방산(g)': 0.3,
    '콜레스테롤(mg)': 0.2,
    '칼륨(mg)': 0.2,
    '칼슘(mg)': 0.4,
    '식이섬유(g)': 0.4    
}

# def compute_weight_per_nutrient(health_index: float, b_values: dict) -> dict:
#     """
#     health_index와 성분별 b값을 곱하여 가중치 계산
#     """
#     return {
#         nutrient: round(health_index * b, 4)
#         for nutrient, b in b_values.items()
#     }

# def compute_deviation_ratio(value, standard, tolerance=0.1):
#     """
#     기준에서 얼마나 벗어났는지를 비율로 계산.
#     허용 오차(tolerance)를 넘을 때만 감점.
#     """
#     lower_bound = standard * (1 - tolerance)
#     upper_bound = standard * (1 + tolerance)

#     if lower_bound <= value <= upper_bound:
#         return 0  # 허용 범위 이내면 감점 없음
#     else:
#         return abs(value - standard) / standard  # 기준 대비 벗어난 비율
    
# def calculate_health_score(food, dash_standard, nutrient_weights, nutrients, tolerance=0.1):
#     base_score = 100
#     total_penalty = 0
#     nutrient_penalties = {}

#     # risky_nutrients = ['나트륨(mg)', '포화지방산(g)', '콜레스테롤(mg)']
#     # deficient_nutrients = ['식이섬유(g)', '칼륨(mg)', '칼슘(mg)']

#     for nutrient in nutrients:
#         value = food[nutrient]
#         standard = dash_standard[nutrient]
#         weight = nutrient_weights[nutrient]
#         deviation_ratio = compute_deviation_ratio(value, standard, tolerance)

#         if deviation_ratio == 0:
#             penalty = 0
#         else:
#             penalty = np.sqrt(deviation_ratio) * weight

#         nutrient_penalties[nutrient] = round(penalty * 100, 2)
#         total_penalty += penalty

#     health_score = max(0, base_score - (total_penalty * 50))
#     return round(health_score, 2), nutrient_penalties

def adjust_dash_standard(bmr: float) -> dict:
    """
    사용자의 BMR에 따라 DASH 기준을 개인화해서 반환 (단위 정확히 맞춤).
    """
    scale = bmr / 2100  # 전체 기준 비례 조정

    return {
        '나트륨(mg)': 2300 * scale,  # mg 단위도 비례 가능
        # '지방(g)': (bmr * 0.27) / 9,
        # '단백질(g)': (bmr * 0.18) / 4,
        # '탄수화물(g)': (bmr * 0.55) / 4,
        '포화지방산(g)': (bmr * 0.06) / 9,
        '콜레스테롤(mg)': 150 * scale,
        '칼륨(mg)': 4700 * scale,
        '칼슘(mg)': 1250 * scale,
        '식이섬유(g)': 30 * scale
    }
    


nutrient_directions = {
    '나트륨(mg)': 'high',
    '포화지방산(g)': 'high',
    '콜레스테롤(mg)': 'high',
    '식이섬유(g)': 'low',
    '칼륨(mg)': 'low',
    '칼슘(mg)': 'low'
}

def compute_penalty_with_direction(value, standard, weight, optimal_range=(0.3, 0.65), direction='high'):
    """
    성분별 '많을수록 나쁨' or '적을수록 나쁨'을 구분하여 감점 계산
    - direction: 'high' = 초과 위험성분 (나트륨 등)
                 'low'  = 결핍 보완성분 (식이섬유 등)
    """
    ratio = value / standard

    if direction == 'high':
        if ratio <= optimal_range[0]:
            return 0.0
        elif ratio <= optimal_range[1]:
            return (ratio - optimal_range[0]) ** 1.5 * weight
        else:
            return (np.exp(ratio - optimal_range[1]) - 1) * weight

    elif direction == 'low':
        if ratio >= optimal_range[1]:
            return 0.0
        elif ratio >= optimal_range[0]:
            return (optimal_range[1] - ratio) ** 1.5 * weight
        else:
            return (np.exp(optimal_range[0] - ratio) - 1) * weight

    else:
        raise ValueError("direction must be 'high' or 'low'")


def calculate_health_score_flexible(food, dash_standard, nutrient_weights, nutrient_directions, optimal_range=(0.4, 0.7)):
    """
    위험 성분('high')과 결핍 성분('low') 모두 포함한 유연한 점수 계산 함수
    """
    base_score = 100
    total_penalty = 0
    nutrient_penalties = {}

    for nutrient, direction in nutrient_directions.items():
        value = food.get(nutrient, 0)
        standard = dash_standard[nutrient]
        weight = nutrient_weights[nutrient]

        penalty = compute_penalty_with_direction(value, standard, weight, optimal_range, direction)
        nutrient_penalties[nutrient] = round(penalty * 100, 2)
        total_penalty += penalty

    health_score = max(0, base_score - (total_penalty * 150))
    return round(health_score, 2), nutrient_penalties

risky_b_values = {
    '나트륨(mg)': 0.5,
    '포화지방산(g)': 0.3,
    '콜레스테롤(mg)': 0.2
}
deficient_b_values = {
    '식이섬유(g)': 0.4,
    '칼륨(mg)': 0.4,
    '칼슘(mg)': 0.2
}

def compute_weight_per_nutrient(health_index, risky_b, deficient_b, risky_ratio=0.7):
    # 위험 성분: health_index ↑ → 감점 ↑
    risky_weights = {
        k: round((health_index ** 2) * b * risky_ratio, 4)
        for k, b in risky_b.items()
    }

    # 결핍 성분: health_index ↑ → 감점 ↓
    deficient_weights = {
        k: round(((1 - health_index) ** 2) * b * (1 - risky_ratio), 4)
        for k, b in deficient_b.items()
    }

    return {**risky_weights, **deficient_weights}

def recommend_foods(user_bmr, health_index, food_df, top_n=5, lowest_n=5): # lowest_n 인자 추가
    # liked_foods, disliked_foods 인자는 현재 사용하지 않으므로, 이 함수 내에서 관련 로직을 제거해도 무방합니다.
    # 만약 나중에 좋아요/싫어요 기능을 활성화할 예정이라면 그대로 두세요.

    dash_standard = adjust_dash_standard(user_bmr)
    nutrient_weights = compute_weight_per_nutrient(health_index, risky_b_values, deficient_b_values)

    food_scores = []
    for _, row in food_df.iterrows():
        food_name = row['식품명']

        # 좋아요/싫어요 기능 미사용 시 이 부분은 주석 처리 또는 제거
        # if disliked_foods and food_name in disliked_foods:
        #     continue

        score, penalties = calculate_health_score_flexible(row, dash_standard, nutrient_weights, nutrient_directions)

        # 좋아요/싫어요 기능 미사용 시 이 부분은 주석 처리 또는 제거
        # if liked_foods and food_name in liked_foods:
        #     score += 5
        #     score = min(100, score)

        food_scores.append({
            '식품명': food_name,
            'health_score': score,
            'penalties': penalties,
            '식품분류': row.get('식품대분류명', '기타'), # 식품대분류명 또는 식품분류 컬럼 사용
            '나트륨(mg)': row.get('나트륨(mg)', 0),
            '에너지(kcal)': row.get('에너지(kcal)', 0)
            # 필요한 다른 영양소 정보도 food_item 딕셔너리에 추가할 수 있습니다.
        })

    # 모든 식품 점수를 계산한 후, 정렬하여 상위 N개와 하위 N개를 모두 추출
    # 1차 정렬: 건강 점수 높은 순 (내림차순)
    sorted_all_foods_desc = sorted(food_scores, key=lambda x: x['health_score'], reverse=True)

    # 상위 N개 (top_n) 추천 식품
    top_recommendations = sorted_all_foods_desc[:top_n]

    # 최하위 N개 (lowest_n) 식품
    # 전체 목록을 낮은 점수 순으로 정렬 (오름차순)하여 앞에서부터 N개 선택
    lowest_recommendations = sorted(food_scores, key=lambda x: x['health_score'], reverse=False)[:lowest_n]


    return top_recommendations, lowest_recommendations # 두 개의 목록 반환