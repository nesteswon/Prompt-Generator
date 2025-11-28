import streamlit as st
import google.generativeai as genai
import openai

# ==============================================================================
# [1] System Instruction (역할 + 규칙)
# ==============================================================================
SYSTEM_INSTRUCTION = """
당신은 입력된 내용을 분석하여 'Flow JSON 프롬프트'와 '미드저니 프롬프트'로 변환하는 전문 AI입니다.
사용자가 입력한 내용을 바탕으로 아래의 [조건]과 [양식]을 완벽하게 준수하여 답변하세요.

[조건 1: Flow JSON 작성]
- 입력된 내용을 바탕으로 JSON의 "___" 부분을 영문으로 번역하여 채우세요.
- JSON 구조(Key값)를 절대 변경하거나 삭제하지 마세요.

[조건 2: 누락 데이터 처리]
- 입력 내용에서 찾을 수 없는 정보는 "none"이라고 기입하세요.
- JSON 작성 후, 하단에 'Flow 사용 json 프롬프트 중 누락 / none 부분'을 별도로 정리하세요.

[조건 3: 미드저니 프롬프트 작성]
- 모든 내용은 영문으로 번역되어야 합니다.
- 다음 순서를 반드시 지켜서 조합하세요:
  주제 → 액션 → 배경 → 카메라 → 스타일 → 구도

[조건 4: 미드저니 누락 확인]
- 누락 요소는 마지막에 리스트로 출력하세요.
"""

# ==============================================================================
# [2] Streamlit UI 설정
# ==============================================================================
st.set_page_config(
    page_title="Flow + Midjourney Prompt Generator",
    layout="wide"
)

st.title("Flow JSON + Midjourney 프롬프트 생성기")
st.caption("Google 또는 OpenAI 모델을 선택하여 프롬프트 생성")

with st.sidebar:
    st.subheader("🔐 API 설정")

    google_api_key = st.text_input(
        "Google API Key (Gemini)",
        type="password",
        placeholder="AIza..."
    )

    openai_api_key = st.text_input(
        "OpenAI API Key (GPT)",
        type="password",
        placeholder="sk-..."
    )

    model_choice = st.selectbox(
        "사용할 모델 선택",
        ["Google Gemini (Flash)", "OpenAI GPT-4.1 / 4o"]
    )

st.markdown("### 1. 변환할 내용을 한국어로 입력하세요.")

default_text = "아침 햇살 아래 카페 테라스에서 노트북을 사용하는 한국인 여성"

user_input = st.text_area(
    "설명 입력",
    value=default_text,
    height=200,
)

generate_btn = st.button("🚀 프롬프트 생성하기")

# ==============================================================================
# [3] LLM 요청 함수
# ==============================================================================

def ask_google(prompt, api_key):
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
    "gemini-2.5-flash",
    system_instruction=SYSTEM_INSTRUCTION
)

    response = model.generate_content(prompt)

    return response.text


def ask_openai(prompt, api_key):
    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # 비용: 매우 저렴, 속도 빠름
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content


# ==============================================================================
# [4] 실행 로직
# ==============================================================================
if generate_btn:
    if not user_input.strip():
        st.error("내용을 입력해 주세요.")

    else:
        try:
            with st.spinner("AI가 프롬프트를 생성 중입니다..."):

                if model_choice == "Google Gemini (Flash)":
                    if not google_api_key:
                        st.error("Google API Key를 입력하세요.")
                        st.stop()
                    result = ask_google(user_input, google_api_key)

                elif model_choice == "OpenAI GPT-4.1 / 4o":
                    if not openai_api_key:
                        st.error("OpenAI API Key를 입력하세요.")
                        st.stop()
                    result = ask_openai(user_input, openai_api_key)

            st.success("프롬프트 생성 완료!")

            # UI 2단 컬럼
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🧩 분석 결과 (Markdown 그대로)")
                st.markdown(result)

            with col2:
                st.markdown("### 📋 Raw Text")
                st.code(result)

        except Exception as e:
            st.error(f"오류 발생: {e}")
