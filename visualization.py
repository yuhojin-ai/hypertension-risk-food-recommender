import streamlit as st
import pandas as pd
import numpy as np
from food_recommender import (
    adjust_dash_standard, compute_weight_per_nutrient, 
    calculate_health_score_flexible, nutrient_directions, 
    risky_b_values, deficient_b_values
)

# 식품 데이터 로딩 (캐싱 적용)
@st.cache_data
def load_food_data():
    return pd.read_csv("data/raw/combined_filterd_fd.csv")

# 식품 점수 분포 시각화
def visualize_food_score_distribution(user_bmr, health_index):
    food_df = load_food_data()
    dash_standard = adjust_dash_standard(user_bmr)
    nutrient_weights = compute_weight_per_nutrient(health_index, risky_b_values, deficient_b_values)

    scores = []
    for _, row in food_df.iterrows():
        score, _ = calculate_health_score_flexible(row, dash_standard, nutrient_weights, nutrient_directions)
        scores.append(score)

    score_df = pd.DataFrame({"health_score": scores})
    st.markdown("### 전체 식품 건강 점수 분포")
    st.bar_chart(score_df["health_score"].value_counts().sort_index())

# 식품 점수 통계 요약
def summarize_food_scores(user_bmr, health_index):
    food_df = load_food_data()
    dash_standard = adjust_dash_standard(user_bmr)
    nutrient_weights = compute_weight_per_nutrient(health_index, risky_b_values, deficient_b_values)

    scores = [
        calculate_health_score_flexible(row, dash_standard, nutrient_weights, nutrient_directions)[0]
        for _, row in food_df.iterrows()
    ]

    if not scores:
        st.warning("식품 점수를 계산할 수 없습니다.")
        return

    score_series = pd.Series(scores)
    stats = score_series.describe()[['min', '25%', '50%', '75%', 'max']]
    stats['mean'] = score_series.mean()
    stats = stats.rename({
        'min': '최소값',
        '25%': '1사분위수(Q1)',
        '50%': '중앙값(Q2)',
        '75%': '3사분위수(Q3)',
        'max': '최댓값',
        'mean': '평균값'
    })

    st.markdown("### 식품 점수 요약 통계")
    st.dataframe(stats.to_frame(name='값'))

# 추천 결과 카드형 시각화 (추천 리스트 받아서 출력)
def display_food_recommendations(recommended):
    st.markdown("### 추천 식품 상세 분석")

    cols = st.columns(len(recommended))
    for col, food in zip(cols, recommended):
        with col:
            st.markdown(f"""
            <div style='border:1px solid #ddd; border-radius:10px; padding:15px; background-color:#f9f9f9'>
                <h4 style='color:#2E86C1'>{food['식품명']}</h4>
                <p><b>점수: {food['health_score']}점</b></p>
                <p>주요 감점 성분:</p>
            """, unsafe_allow_html=True)

            top_penalty = sorted(food['penalties'].items(), key=lambda x: x[1], reverse=True)[:3]
            for nutrient, penalty in top_penalty:
                st.write(f"- {nutrient}: -{penalty:.2f}")

            st.markdown("</div>", unsafe_allow_html=True)
