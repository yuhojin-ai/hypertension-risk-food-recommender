# health_report.py
import streamlit as st
import pandas as pd
import plotly.express as px # 시각화를 위해 추가

# BMI 범주 및 색상 정의 함수
def get_bmi_category_and_color(bmi):
    if bmi < 18.5:
        return "저체중", "#3498db" # Blue
    elif 18.5 <= bmi < 23:
        return "정상", "#2ecc71" # Green
    elif 23 <= bmi < 25:
        return "과체중", "#f1c40f" # Yellow
    elif 25 <= bmi < 30:
        return "경도 비만", "#e67e22" # Orange
    else: # BMI >= 30
        return "고도 비만", "#e74c3c" # Red

def display_user_profile(age, sex, height, weight, bmi):
    st.subheader("🧍 사용자 프로필")
    
    bmi_category, bmi_color = get_bmi_category_and_color(bmi)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("성별", sex)
    with col2:
        st.metric("나이", f"{age}세")
    with col3:
        st.metric("BMI", f"{bmi:.1f}")
        st.markdown(f"<small style='color:{bmi_color};'><b>{bmi_category}</b></small>", unsafe_allow_html=True)
        # BMI 바 (선택적 시각화)
        st.progress(min(bmi / 40, 1.0)) # BMI 40을 최대값으로 가정, 1.0 초과 방지

def get_qualitative_assessment(score_name, score_value):
    """각 점수별 질적 평가와 색상을 반환하는 함수"""
    # 각 점수의 특성에 맞게 기준치와 평가 문구, 색상 조정 필요
    if score_name == "고혈압 발병 확률": # 낮을수록 좋음
        if score_value <= 0.4: return "양호", "green"
        elif score_value <= 0.65: return "주의", "orange"
        elif score_value <= 0.85: return "위험", "red"
        else: return "고위험", "darkred"
    elif score_name == "건강 위험 지수": # 낮을수록 좋음 (0~1 범위 가정)
        if score_value <= 0.3: return "낮음 (양호)", "green"
        elif score_value <= 0.6: return "보통", "orange"
        else: return "높음 (주의)", "red"
    elif score_name == "고혈압 위험 점수": # 낮을수록 좋음 (0~1 범위 가정)
        if score_value <= 0.3: return "낮음 (양호)", "green"
        elif score_value <= 0.6: return "보통", "orange"
        else: return "높음 (주의)", "red"
    elif score_name == "신체 에너지 점수": # 높을수록 좋음 (0~1 범위 가정)
        if score_value >= 0.7: return "높음 (양호)", "green"
        elif score_value >= 0.4: return "보통", "orange"
        else: return "낮음 (개선 필요)", "red"
    return "", "black" # 기본값

def display_risk_scores(proba, risk_level_from_proba, risk_score, metabolic_score, health_index): # risk_level 인자명 변경
    st.subheader("📊 건강 주요 지표")
    st.caption("각 지표는 회원님의 건강 상태를 다각도로 분석한 결과입니다. 설명을 통해 각 지수의 의미를 확인해보세요.")

    # 점수 설명 및 해석 정의 (이미지 로직 기반으로 업데이트)
    score_info = {
        "고혈압 발병 확률": {
            "value": proba * 100,
            "unit": "%",
            "risk_level_text": risk_level_from_proba, # 모델 예측 단계에서 계산된 '양호', '주의' 등
            "definition": "머신러닝 모델을 통해 예측된 고혈압 발병 가능성입니다.",
            "interpretation_factors": "다양한 건강검진 수치들을 종합적으로 고려하여 계산됩니다.",
            "meaning_for_user": "이 확률이 높을수록 고혈압 예방 및 관리에 더 많은 주의가 필요함을 의미합니다.",
            "ideal_direction_text": "낮을수록 좋습니다."
        },
        "건강 위험 지수": { # Health Index (이미지: 고혈압 위험 점수 * 0.7 + (1-신체에너지 점수) * 0.3)
            "value": health_index,
            "unit": "점",
            "definition": "고혈압 관련 위험 요인과 신체 에너지 상태를 종합적으로 고려한 전반적인 건강 위험 수준입니다.",
            "interpretation_factors": "'고혈압 위험 점수'가 높거나 '신체 에너지 점수'가 낮을수록 이 지수가 높아집니다.",
            "meaning_for_user": "이 지수가 높을수록 식단 관리 시 건강 개선을 위한 요소가 더 많이 고려됩니다.",
            "ideal_direction_text": "낮을수록 좋습니다."
        },
        "고혈압 위험 점수": { # Risk Score (이미지: 여러 건강 지표를 정규화하고 가중치를 곱한 점수)
            "value": risk_score,
            "unit": "점",
            "definition": "혈액검사 수치 등 고혈압 발병에 영향을 미치는 주요 건강 지표들을 통합하여 평가한 점수입니다.",
            "interpretation_factors": "BMI, 혈당, 콜레스테롤, 혈압 관련 지표 등이 기준치에서 벗어날수록 높아질 수 있습니다.",
            "meaning_for_user": "이 점수는 어떤 건강 지표 관리가 필요한지 파악하는 데 도움을 줍니다.",
            "ideal_direction_text": "낮을수록 좋습니다."
        },
        "신체 에너지 점수": { # Metabolic Score (이미지: BMI 정규화*0.6 + BMR 정규화*0.4)
            "value": metabolic_score,
            "unit": "점",
            "definition": "체질량지수(BMI)와 기초대사량(BMR)을 바탕으로 평가한 현재 신체의 대사 효율성 및 에너지 수준입니다.",
            "interpretation_factors": "적정 체중(BMI)을 유지하고 기초대사량이 상대적으로 높을수록 이 점수가 향상됩니다.", # 이미지 기준
            "meaning_for_user": "이 점수가 높을수록 신체 대사가 원활하고 활력이 좋은 상태로 볼 수 있습니다.",
            "ideal_direction_text": "높을수록 좋습니다."
        }
    }

    cols = st.columns(2) # 2열로 배치

    for i, (score_name, info) in enumerate(score_info.items()):
        col = cols[i % 2]
        with col:
            # 질적 평가 및 색상 가져오기
            # '고혈압 발병 확률'은 risk_level_from_proba를 직접 사용하거나 get_qualitative_assessment에서 proba 값으로 평가
            if score_name == "고혈압 발병 확률":
                qual_text = info['risk_level_text']
                # 색상은 risk_level_from_proba에 따라 직접 지정하거나 get_qualitative_assessment 재활용
                if qual_text == "양호": qual_color = "green"
                elif qual_text == "주의": qual_color = "orange"
                elif qual_text == "위험": qual_color = "red"
                else: qual_color = "darkred" # 고위험
            else:
                qual_text, qual_color = get_qualitative_assessment(score_name, info['value'])

            # 메트릭 표시
            delta_text_display = f"{qual_text} ({info['ideal_direction_text']})"
            if score_name == "고혈압 발병 확률":
                st.metric(
                    label=score_name,
                    value=f"{info['value']:.2f}{info['unit']}",
                    delta=qual_text, # '양호', '주의' 등
                    delta_color="off" # 사용자 정의 색상을 위해 기본 델타 색상 끔
                )
                st.markdown(f"<span style='color:{qual_color}; font-size:small;'><b>{qual_text}</b> ({info['ideal_direction_text']})</span>", unsafe_allow_html=True)

            else:
                st.metric(
                    label=score_name,
                    value=f"{info['value']:.3f}{info['unit']}",
                    delta=qual_text, # 질적 평가 (예: "높음 (양호)")
                    delta_color="off" # 사용자 정의 색상을 위해 기본 델타 색상 끔
                )
                st.markdown(f"<span style='color:{qual_color}; font-size:small;'><b>{qual_text}</b> ({info['ideal_direction_text']})</span>", unsafe_allow_html=True)


            # 점수 시각화 (st.progress 또는 사용자 정의 바)
            # 0-1 범위로 정규화된 값이라고 가정 (고혈압 확률은 0-100%를 0-1로 변환)
            progress_value = info['value'] / 100 if score_name == "고혈압 발병 확률" else info['value']
            if "낮을수록 좋" in info['ideal_direction_text']: # 낮을수록 좋은 지표 (빨간색으로 진행)
                # st.progress는 항상 초록색이므로, 직접 HTML로 바를 만들거나,
                # 값에 따라 색이 변하는 텍스트로 대체하거나, Plotly 같은 라이브러리 사용 필요.
                # 여기서는 간단히 st.progress를 사용하되, 의미 해석에 주의를 줌.
                # st.progress(progress_value) # 초록색 바로 표시됨
                # 아니면 색상있는 텍스트로 대체
                pass # st.metric 아래 markdown으로 질적 평가를 색상으로 표시했으므로 progress는 생략 가능
            else: # 높을수록 좋은 지표 (초록색으로 진행)
                # st.progress(progress_value)
                pass

            with st.popover("ℹ️ 더 알아보기", use_container_width=False):
                st.markdown(f"##### {score_name}")
                st.markdown(f"**🔹 무엇인가요?**\n{info['definition']}")
                st.markdown(f"**🔹 어떻게 해석하나요?**\n{info['ideal_direction_text']} 현재 회원님의 점수는 **<span style='color:{qual_color};'>{qual_text}</span>** 수준입니다.", unsafe_allow_html=True)
                st.markdown(f"**🔹 주요 영향 요인?**\n{info['interpretation_factors']}")
                st.markdown(f"**🔹 그래서 무엇을 의미하나요?**\n{info['meaning_for_user']}")
            
def display_contributing_factors(contributions, user_bmi, user_age):
    st.subheader("🔬 주요 영향 인자 분석")

    if not contributions:
        st.info("분석된 주요 영향 인자가 없습니다.")
        return

    sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    
    chart_contrib_items = []
    text_explanations = []

    AGE_FACTOR_KEY = '연령대코드(5세단위)'
    BMI_FACTOR_KEY = 'bmi'

    for factor, value in sorted_contrib:
        if value <= 0.001: 
            continue

        if factor == AGE_FACTOR_KEY:
            text_explanations.append(
                f"**연령 ({user_age}세)**: "
                f"나이는 고혈압 위험도에 자연스럽게 영향을 미치는 요소입니다 (모델 기여도: {value:.3f}). "
                "이는 변경할 수 없는 요인이지만, 다른 생활 습관 관리를 통해 전반적인 위험을 낮출 수 있습니다."
            )
        elif factor == BMI_FACTOR_KEY:
            bmi_category, _ = get_bmi_category_and_color(user_bmi)
            
            # BMI 상태에 따라 다른 설명 생성
            bmi_intro_text = f"**체질량지수(BMI)**: 현재 {user_bmi:.1f} ({bmi_category}) 입니다. "
            bmi_model_contrib_text = (
                f"저희 분석 모델에서는 BMI가 고혈압 위험도에 상대적으로 영향을 주는 요인으로 나타났습니다 (모델 기여도: {value:.3f}). "
            )

            if "정상" in bmi_category: # BMI가 정상 범위인 경우
                bmi_advice_text = (
                    "이는 현재 다른 건강 지표들이 매우 안정적이거나, 모델이 BMI 변화에 민감하게 반응하기 때문일 수 있습니다. "
                    "따라서 현재의 건강한 체중을 꾸준히 유지하시는 것이 중요합니다."
                )
            else: # BMI가 정상 범위가 아닌 경우 (저체중, 과체중, 비만 등)
                bmi_advice_text = (
                    f"이 수치는 **{bmi_category}** 상태로 개선이 필요합니다. "
                    "적극적인 체중 관리를 통해 BMI를 건강 범위로 낮추는 것이 고혈압 위험 감소에 매우 중요합니다."
                )
            
            text_explanations.append(bmi_intro_text + bmi_model_contrib_text + bmi_advice_text)

        else:
            chart_contrib_items.append((factor, value))

    if text_explanations:
        st.markdown("##### 연령 및 BMI의 영향에 대하여")
        for explanation in text_explanations:
            st.markdown(f"• {explanation}", unsafe_allow_html=True)
        st.markdown("---")

    if not chart_contrib_items:
        st.info("현재 혈액검사 수치 등 다른 주요 인자들은 비교적 안정적인 상태로 보입니다.")
    else:
        st.markdown("##### 생활습관으로 개선 가능한 주요 영향 인자 (Top 5)")
        contrib_df_chart = pd.DataFrame(chart_contrib_items, columns=['인자', '기여도 점수'])
        
        top_n = 5
        contrib_df_top_n = contrib_df_chart.head(top_n)

        if not contrib_df_top_n.empty:
            fig = px.bar(
                contrib_df_top_n,
                x='기여도 점수',
                y='인자',
                orientation='h',
                labels={'기여도 점수':'위험도 기여 점수', '인자':'건강 지표'},
                color='기여도 점수',
                color_continuous_scale=px.colors.sequential.Reds
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
             st.info("현재 혈액검사 수치 등 다른 주요 인자들은 비교적 안정적인 상태로 보입니다.")

    with st.expander("모든 인자별 상세 기여도 보기 (연령, BMI 포함)"):
        if sorted_contrib:
            for factor, value in sorted_contrib:
                if value > 0.001:
                    st.write(f"- {factor}: {value:.3f}")
        else:
            st.write("표시할 기여도 정보가 없습니다.")


def display_management_tip(contributions, user_bmi): # 함수 인자 이름을 user_bmi로 명확히 함
    st.subheader("💡 맞춤형 개선 가이드")

    if not contributions:
        st.success("🎉 현재 특별히 관리할 위험 요소는 보이지 않습니다. 건강한 생활을 유지하세요!")
        return

    # 기여도가 가장 높은 인자 찾기
    # contributions 딕셔너리가 비어있지 않다고 가정 (위에서 체크함)
    top_issue_factor, top_issue_value = max(contributions.items(), key=lambda x: x[1])

    management_messages = []
    message_type = "info" # 메시지 타입을 info, warning, success 등으로 관리

    # config.py 또는 model_predict.py 등에서 사용하는 정확한 키 이름으로 설정해주세요.
    # 이 키는 'contributions' 딕셔너리의 키와 일치해야 합니다.
    AGE_FACTOR_KEY = '연령대코드(5세단위)'
    BMI_FACTOR_KEY = 'bmi'

    # 가장 큰 기여도 값 자체가 매우 낮은 경우 (예: 0.05 이하)
    if top_issue_value <= 0.05:
        management_messages.append("🎉 전반적으로 모든 건강 지표가 잘 관리되고 있는 것으로 보입니다. 현재의 건강한 생활 습관을 꾸준히 유지해주세요!")
        message_type = "success"
    
    # 가장 큰 기여 인자가 BMI인 경우
    elif top_issue_factor == BMI_FACTOR_KEY:
        bmi_category, _ = get_bmi_category_and_color(user_bmi)
        
        management_messages.append(
            f"저희 분석에 따르면, **'{top_issue_factor}'**가 현재 회원님의 고혈압 위험도에 상대적으로 가장 큰 영향을 주는 요인으로 나타났습니다 (모델 기여도: {top_issue_value:.3f})."
        )
        management_messages.append(f"현재 회원님의 BMI는 **{user_bmi:.1f} ({bmi_category})** 입니다.")

        if "정상" in bmi_category: # BMI가 정상 범위인 경우
            message_type = "success" # 긍정적/유지 메시지 톤
            management_messages.append(
                "매우 건강한 BMI를 유지하고 계십니다! 👍 이처럼 좋은 상태를 꾸준히 유지하는 것이 고혈압 예방 및 전반적인 건강 관리에 매우 중요합니다."
            )
            management_messages.append("**건강 체중 유지를 위한 권장 사항:**")
            management_messages.append("  - **균형 잡힌 식단 지속**: 다채로운 채소와 과일, 통곡물, 건강한 단백질원을 꾸준히 섭취하여 현재의 건강한 식습관을 이어가세요.")
            management_messages.append("  - **규칙적인 신체 활동 생활화**: 즐겨 하시는 유산소 운동(예: 걷기, 조깅, 자전거)과 근력 운동을 병행하여 신체 기능을 활발하게 유지하세요.")
        else: # BMI가 정상 범위가 아닌 경우 (예: 과체중, 비만 등)
            message_type = "warning"
            management_messages.append(
                f"따라서 **'{top_issue_factor}' 수치를 정상 범위({bmi_category} → 정상)로 개선**하는 것이 현재 가장 중요합니다."
            )
            management_messages.append("**체중 관리를 위한 적극적인 노력:**")
            management_messages.append("  - **식단 조절**: 섭취 칼로리를 조절하고, 가공식품 및 고당분, 고지방 음식 섭취를 줄이세요. 섬유질이 풍부한 채소와 양질의 단백질 위주로 식단을 구성하는 것이 좋습니다.")
            management_messages.append("  - **운동량 증가**: 현재 하고 계신 운동의 강도나 시간을 점진적으로 늘리거나, 새로운 활동을 추가해보세요. 일상생활에서의 활동량(예: 계단 이용)을 늘리는 것도 도움이 됩니다.")
            management_messages.append("  - 필요하다면 의사 또는 영양사와 상담하여 개인 맞춤형 계획을 세우는 것을 권장합니다.")
            
    # 가장 큰 기여 인자가 '나이'인 경우
    elif top_issue_factor == AGE_FACTOR_KEY:
        message_type = "info" # 나이는 직접적 '경고' 대상이 아님
        management_messages.append(
            f"**'{top_issue_factor}'**는 고혈압 위험도에 자연스럽게 영향을 미치는 요인 중 하나입니다 (모델 기여도: {top_issue_value:.3f})."
        )
        management_messages.append(
            "나이는 조절할 수 없는 부분이지만, 다른 생활 습관 요인들(식단, 운동, 스트레스 관리 등)을 건강하게 관리함으로써 전반적인 고혈압 위험을 효과적으로 낮추는 것이 중요합니다."
        )
        management_messages.append("균형 잡힌 생활 습관 유지에 더욱 신경 써주시면 좋겠습니다.")

    # 그 외 다른 건강 지표가 가장 큰 기여 인자인 경우
    else:
        message_type = "warning"
        management_messages.append(
            f"⚠ 가장 주의가 필요한 항목은 **'{top_issue_factor}'**입니다 (모델 기여도: {top_issue_value:.3f}). 이 수치를 개선하기 위한 노력이 필요합니다."
        )
        # 특정 인자별 상세 팁 추가 (예시)
        if top_issue_factor == '식전혈당(공복혈당)':
            management_messages.append("  - **혈당 관리 Tip**: 정제된 탄수화물(흰쌀밥, 밀가루, 설탕 등) 섭취를 줄이고, 혈당 지수가 낮은 복합 탄수화물(현미, 통밀, 채소 등)을 선택하세요. 규칙적인 식사 시간을 지키고, 과식을 피하는 것이 중요합니다.")
        elif top_issue_factor == '트리글리세라이드':
            management_messages.append("  - **중성지방 관리 Tip**: 과도한 음주와 단순당(설탕, 과당 등) 섭취를 피하고, 트랜스지방이 많은 가공식품을 줄이세요. 오메가-3 지방산이 풍부한 등푸른 생선, 견과류 등을 섭취하는 것이 도움이 될 수 있습니다.")
        elif top_issue_factor == 'HDL콜레스테롤': # HDL은 높을수록 좋음, 기여도가 높다는 것은 낮아서 문제라는 의미일 수 있음 (모델 해석에 따라 다름)
             management_messages.append(f"  - '{top_issue_factor}' 수치가 낮아 위험 요인으로 작용하고 있을 수 있습니다. 유산소 운동과 건강한 지방 섭취를 통해 수치를 높이는 것이 좋습니다.")
        # 다른 인자들에 대한 구체적인 팁 추가 가능
        else:
            management_messages.append(f"  - '{top_issue_factor}' 수치 개선을 위해 관련 건강 정보를 찾아보거나, 필요시 전문가와 상담하는 것을 권장합니다.")

    # 최종 메시지 출력
    if message_type == "success":
        st.success("\n\n".join(management_messages)) # 문단 구분을 위해 \n\n 사용
    elif message_type == "warning":
        st.warning("\n\n".join(management_messages))
    else: # info 또는 기타
        st.info("\n\n".join(management_messages))

    # 생활 습관 개선을 위한 추가 팁 (st.expander)은 이전과 동일하게 유지 가능
    with st.expander("✍️ 건강한 생활 습관을 위한 추가 팁 보기"):
        st.markdown("""
        * **균형 잡힌 식단**: 매일 다채로운 채소와 과일을 충분히 섭취하고, 통곡물, 저지방 단백질 위주로 식단을 구성하세요.
        * **나트륨 섭취 줄이기**: 국물 섭취를 줄이고, 가공식품 및 외식 시 나트륨 함량을 확인하며 싱겁게 먹는 습관을 들이세요.
        * **규칙적인 신체 활동**: 일주일에 최소 150분 이상 중강도 유산소 운동(빠르게 걷기 등) 또는 75분 이상 고강도 유산소 운동을 실천하세요. 주 2회 이상 근력 운동도 좋습니다.
        * **금연은 필수, 절주는 권장**: 흡연은 모든 심혈관 질환의 강력한 위험 요인입니다. 과도한 음주 또한 혈압에 부정적인 영향을 미칩니다.
        * **충분한 수면과 스트레스 관리**: 매일 7-8시간의 질 좋은 수면을 취하고, 명상, 취미 활동 등으로 스트레스를 효과적으로 관리하세요.
        """)