import streamlit as st
from openai import OpenAI

# 🔐 Streamlit Secrets 에서 OpenAI API Key 가져오기
OPENAI_API_KEY = st.secrets["openai_api_key"]

# ==============================================================================
# [1] System Instruction 설정 (역할 + 규칙)
# ==============================================================================
SYSTEM_INSTRUCTION = """
당신은 입력된 내용을 분석하여 'ComfyUI JSON 프롬프트'와 '미드저니 프롬프트'로 변환하는 전문 AI입니다.
사용자가 입력한 내용을 바탕으로 아래의 [조건]과 [양식]을 완벽하게 준수하여 답변하세요.

[조건 1: ComfyUI JSON 작성]
- 입력된 내용을 바탕으로 JSON의 "___" 부분을 영문으로 번역하여 채우세요.
- JSON 구조(Key값)를 절대 변경하거나 삭제하지 마세요.
- 입력으로 제공되는 [오디오 / 사운드], [타임라인 / 씬 분할] 정보는 반드시 JSON의 "audio"와 "timeline" 섹션에 반영해야 합니다.
- 카메라 정보(샷 타입, 카메라 움직임, 화면 내용)는 가능한 한 timeline.scenes 안에 통합해서 표현하세요.
- camera_work 섹션은 전체 영상에 공통으로 적용되는 렌즈, 효과, 전역적인 카메라 스타일 정도만 간단히 채우세요.
- 카메라, 렌즈, 조명 등은 설명이 없을 때는 당신이 장면에 어울리는 값을 추천하여 채워 주세요.

[조건 2: 누락 데이터 처리]
- 입력 내용에서 찾을 수 없는 정보는 "none"이라고 기입하세요.
- 카메라, 렌즈, 조명 등은 설명이 없을 떄는 추천으로 채우고, 정말 결정하기 어려운 경우에만 "none"을 사용하세요.
- JSON 작성 후, 하단에 'ComfyUI 사용 json 프롬프트 중 누락 / none 부분'을 별도로 정리하세요.

[조건 3: 미드저니 프롬프트 작성]
- 모든 내용은 영문으로 번역되어야 합니다.
- 다음 순서를 반드시 지켜서 조합하세요:
  주제(Topic) → 액션(Action) → 배경(Background) → 카메라 움직임(Camera) → 스타일(Style) → 구도(Composition)
- 각 요소는 쉼표(,)로 구분하세요.
- 오디오 / 타임라인에서 유추되는 분위기나 리듬감이 있다면, Style / Camera movement / Mood에 자연스럽게 녹여서 표현하세요.

[조건 4: 미드저니 누락 확인]
- 미드저니 프롬프트 작성 후, 부족하거나 빠진 요소를 하단에 정리하세요.

[출력 양식]
1️⃣ ComfyUI 사용 json 프롬프트
- JSON 코드 블럭 형식으로 출력

⚠️ ComfyUI 사용 json 프롬프트 중 누락 / none 부분
- 누락된 항목 목록을 마크다운 리스트로 출력

2️⃣ 미드저니 사용 프롬프트
- 한 줄짜리 영문 프롬프트로 출력
- 구성 순서: 주제, 액션, 배경, 카메라 움직임, 스타일, 구도 (각 요소는 쉼표로 구분)

⚠️ 미드저니 사용 프롬프트 중 누락부분
- 부족하거나 빠진 요소를 리스트로 정리

[JSON 템플릿]

{
  "topic_and_content": {
    "description": "___"
  },
  "character": {
    "gender": "___",
    "appearance": {
      "nationality": "___",
      "age": "___",
      "eye_color": "___",
      "scar": "___",
      "hair": "___"
    },
    "clothing": "___",
    "emotions_sequence": [
      "___",
      "___",
      "___"
    ]
  },
  "action": {
    "sequence": [
      "___",
      "___",
      "___",
      "___"
    ],
    "object_interaction": [
      "___",
      "___",
      "___"
    ]
  },
  "background": {
    "location": "___",
    "time_of_day": "___",
    "elements": [
      "___",
      "___"
    ],
    "weather": "___",
    "scene_lighting": "___"
  },
  "camera_work": {
    "lens": "___",
    "effects": [
      "___",
      "___"
    ],
    "notes": "___"
  },
  "style": {
    "genre": "___",
    "style_lighting": "___",
    "film_grain": "___",
    "color_palette": "___",
    "mood": "___"
  },
  "audio": {
    "bgm": "___",
    "sfx": [
      "___",
      "___"
    ],
    "voice_over": "___"
  },
  "timeline": {
    "overview": "___",
    "scenes": [
      {
        "start_time": 0,
        "end_time": 3,
        "shot_type": "___",
        "camera_movement": "___",
        "description": "___"
      },
      {
        "start_time": 3,
        "end_time": 6,
        "shot_type": "___",
        "camera_movement": "___",
        "description": "___"
      },
      {
        "start_time": 6,
        "end_time": 8,
        "shot_type": "___",
        "camera_movement": "___",
        "description": "___"
      }
    ]
  },
  "aspect_ratio": "___",
  "requirements": "full-size video without letterboxes"
}

[ComfyUI 사용 json 프롬프트 중 누락 / none 부분 예시]

1. character.appearance.eye_color : 눈 색상 정보 없음
2. character.appearance.scar : 흉터 유무 정보 없음

[미드저니 사용 프롬프트 출력 예시]

Topic, Action, Background, Camera movement, Style, Composition

[미드저니 사용 프롬프트 중 누락부분 예시]

1. 카메라 움직임 관련 구체적인 표현 부족
2. 조명 스타일 구체 정보 부족
"""

# ==============================================================================
# [2] Streamlit UI
# ==============================================================================
st.set_page_config(page_title="ComfyUI + Midjourney Prompt Converter (GPT)", layout="wide")

st.title("ComfyUI JSON + Midjourney 프롬프트 변환기 (OpenAI 전용)")
st.caption("한글 설명 → ComfyUI용 JSON 프롬프트 + 미드저니용 영문 프롬프트 자동 생성")

with st.sidebar:
    st.subheader("🔐 API 설정")
    st.markdown(
        "- OpenAI API Key는 Streamlit Secrets에 `openai_api_key` 로 저장되어 사용됩니다.\n"
        "- 이 화면에서는 별도의 키 입력이 필요 없습니다."
    )
    st.markdown("---")
    st.markdown("**사용 모델:** `gpt-4.1-mini` (원하면 코드에서 변경 가능)")

st.markdown("## 1) 기본 정보")

col1, col2 = st.columns(2)
with col1:
    brand = st.text_input("브랜드 / 프로젝트 이름", value="니코모리")
    prompt_name = st.text_input("프롬프트 이름 (내가 구분용으로 쓸 제목)", value="카페 테라스 작업 씬")

with col2:
    aspect = st.selectbox("영상 비율 (Aspect Ratio)", ["16:9", "9:16", "1:1", "21:9"], index=0)
    duration = st.number_input("영상 길이 (초)", min_value=3, max_value=60, value=8)

st.markdown("---")
st.markdown("## 2) 인물 / 캐릭터 / 액션")

col3, col4 = st.columns(2)
with col3:
    subject = st.text_input("주제 / 메인 인물", value="밝게 미소 짓는 20대 한국인 여성")
    character_detail = st.text_area(
        "캐릭터 디테일 (외모, 헤어, 의상 등)",
        height=100,
        value="긴 생머리, 깔끔한 셔츠와 데님, 자연스러운 메이크업"
    )

with col4:
    action = st.text_area(
        "액션 / 행동 (무엇을 하고 있는지)",
        height=100,
        value="카페 테라스에서 노트북으로 작업하며, 가끔 창밖을 보며 미소 짓는다"
    )
    emotion = st.text_input("감정 / 분위기", value="집중 + 여유 + 작은 설렘")

st.markdown("---")
st.markdown("## 3) 배경 / 카메라 / 스타일")

col5, col6 = st.columns(2)
with col5:
    background = st.text_area(
        "배경 / 장소 설명",
        height=100,
        value="햇살이 들어오는 도심 카페 테라스, 주변에 화분과 나무, 뒤로 흐릿한 도시 풍경"
    )
    lighting = st.text_input("조명 / 분위기", value="golden hour, soft natural light")

with col6:
    camera_move = st.text_input(
        "카메라 움직임 / 샷 타입",
        value="slow dolly-in, medium shot, 약간 높은 앵글"
    )
    style = st.text_input(
        "스타일 (예: 시네마틱, 픽사풍, 사진 스타일 등)",
        value="cinematic, realistic, soft color grading"
    )
    composition = st.text_input(
        "구도 (예: rule of thirds, center framing 등)",
        value="rule of thirds, subject slightly off-center"
    )

st.markdown("---")
st.markdown("## 4) 오디오 / 사운드")

col_a1, col_a2 = st.columns(2)
with col_a1:
    audio_bgm = st.text_input(
        "배경 음악 (BGM)",
        value="warm lo-fi beat, soft piano, medium tempo",
        help="음악 장르, 분위기, 템포 등을 적어주세요."
    )
    audio_sfx = st.text_area(
        "효과음 (SFX)",
        height=80,
        value="카페 사람들 소음, 잔잔한 대화 소리, 컵 부딪히는 소리",
        help="현장감 있는 소리, 환경음 등을 적어주세요."
    )

with col_a2:
    audio_voice = st.text_area(
        "내레이션 / 대사 (선택)",
        height=120,
        placeholder="예: 그녀의 내레이션, 브랜드 메시지, 짧은 카피 문구 등"
    )

st.markdown("---")
st.markdown("## 5) 타임라인 / 씬 분할")

timeline_overview = st.text_input(
    "타임라인 요약",
    value="총 8초, 3개의 주요 구간으로 구성",
    help="전체 길이와 씬 분할 개수 정도를 간단히 적어주세요."
)

timeline_detail = st.text_area(
    "씬별 타임라인 (초 단위로 적어도 좋아요)",
    height=140,
    value=(
        "0-3초: 카페 전경, 테라스와 도시 배경을 보여주는 와이드 샷\n"
        "3-6초: 노트북으로 작업 중인 인물을 중심으로 미디엄 샷, 화면에 집중하는 표정\n"
        "6-8초: 살짝 카메라가 줌인되며 창밖을 보며 미소 짓는 클로즈업"
    ),
    help="0-3초 / 3-6초 처럼 시간대별로 어떤 장면이 나오는지 적어주세요."
)

st.markdown("---")
st.markdown("## 6) 추가 메모")

extra = st.text_area(
    "추가로 반영되면 좋은 요소들 (선택)",
    height=80,
    placeholder="예: 손에 머그컵 들고 있음, 바람에 머리카락이 살짝 흩날림, 브랜딩 컬러를 배경에 살짝 반영 등"
)

generate_btn = st.button("🚀 프롬프트 생성하기")

# ==============================================================================
# [3] OpenAI 호출 함수
# ==============================================================================
def ask_openai(prompt: str) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model = "gpt-4.1-mini",  # 필요하면 gpt-4.1 / gpt-4.1-mini 등으로 변경 가능
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


# ==============================================================================
# [4] 생성 로직
# ==============================================================================
if generate_btn:
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY가 설정되지 않았습니다. Secrets에 'openai_api_key'를 등록해 주세요.")
    else:
        # 1) 세분화된 입력들을 하나의 텍스트로 합치기
        combined_prompt = f"""
[브랜드/프로젝트]
{brand}

[프롬프트 이름]
{prompt_name}

[영상 정보]
- Aspect Ratio: {aspect}
- Duration: {duration}초

[주제 / 메인 인물]
{subject}

[캐릭터 디테일]
{character_detail}

[액션 / 행동]
{action}

[감정 / 분위기]
{emotion}

[배경 / 장소]
{background}

[조명 / 분위기]
{lighting}

[카메라 움직임 / 샷 타입]
{camera_move}

[스타일]
{style}

[구도]
{composition}

[오디오 / 사운드]
{composition}
- BGM: {audio_bgm}
- SFX: {audio_sfx}
- Voice / Narration: {audio_voice}

[타임라인 / 씬 분할]
- 요약: {timeline_overview}
- 상세:
{timeline_detail}

[추가 메모]
{extra}
""".strip()

        try:
            with st.spinner("OpenAI가 프롬프트를 생성하는 중입니다..."):
                result_text = ask_openai(combined_prompt)

            st.success("프롬프트 생성 완료!")

            # 🔹 왼쪽: 전체 결과
            left, right = st.columns(2)

            with left:
                st.markdown("### 🧩 전체 결과 (Markdown)")
                st.markdown(result_text)

            # 🔹 오른쪽: 미드저니 프롬프트만 코드박스로
            with right:
                st.markdown("### 🎨 Midjourney 프롬프트 (코드 복사용)")

                text = result_text

                # 1) 미드저니 섹션 시작 마커들
                start_markers = [
                    "2️⃣ 미드저니 사용 프롬프트",
                    "### 2️⃣ 미드저니 사용 프롬프트",
                    "미드저니 사용 프롬프트"
                ]

                start_index = -1
                for marker in start_markers:
                    if marker in text:
                        start_index = text.index(marker) + len(marker)
                        break

                if start_index == -1:
                    st.info("미드저니 프롬프트 구간을 찾을 수 없습니다.")
                    st.code(result_text, language="text")
                else:
                    # 2) 시작 지점 이후 텍스트만 남기기
                    mj = text[start_index:].strip()

                    # 3) 끝 마커들(누락 리스트/다음 섹션) 전에 자르기
                    end_markers = [
                        "⚠️",   # 누락 리스트 시작
                        "###",  # 새로운 섹션
                        "1️⃣",
                        "3️⃣"
                    ]
                    end_index = len(mj)
                    for end in end_markers:
                        if end in mj:
                            pos = mj.index(end)
                            end_index = min(end_index, pos)

                    mj = mj[:end_index].strip()

                    # 4) 백틱 제거 + 제목 줄 제거
                    mj = mj.replace("```", "").strip()

                    lines = mj.splitlines()
                    if len(lines) > 1:
                        first_line = lines[0]
                        if ("프롬프트" in first_line) or ("Prompt" in first_line):
                            mj = "\n".join(lines[1:]).strip()

                    # 🔥 최종 Midjourney 프롬프트만 출력
                    st.code(mj, language="text")

        except Exception as e:
            st.error(f"실행 중 오류가 발생했습니다: {e}")

st.markdown(
    """
    <hr>
    <p style='text-align:center; color: gray; font-size: 14px;'>
    © 2025 NICO MORI. All rights reserved.
    </p>
    """,
    unsafe_allow_html=True
)
