import streamlit as st
from openai import OpenAI
import base64

# 🔐 Streamlit Secrets 에서 OpenAI API Key 가져오기
OPENAI_API_KEY = st.secrets["openai_api_key"]

# ==============================================================================
# [1] System Instruction 설정 (역할 + 규칙)
# ==============================================================================
SYSTEM_INSTRUCTION = """
당신은 입력된 내용을 분석하여 'ComfyUI JSON 프롬프트'와 '미드저니 프롬프트'로 변환하는 전문 AI입니다.
사용자가 입력한 내용을 바탕으로 아래의 [조건]과 [양식]을 완벽하게 준수하여 답변하세요.

[전체 역할]
- 사용자는 한국어로 장면/인물/배경/오디오/타임라인 정보를 입력합니다.
- 당신은 이를 토대로:
  1) ComfyUI에서 사용할 수 있는 JSON 프롬프트
  2) Midjourney에서 사용할 수 있는 한 줄짜리 영문 프롬프트
  를 생성해야 합니다.

[조건 1: ComfyUI JSON 작성]
- 아래 [JSON 템플릿]의 구조와 key 이름, 계층 구조를 절대 변경하지 마세요.
- "___" 부분을 입력 내용을 기반으로 모두 영어로 채우세요.
- 입력으로 제공되는 [오디오 / 사운드], [타임라인 / 씬 분할] 정보는 반드시 JSON의 "audio"와 "timeline" 섹션에 반영해야 합니다.
- "timeline" 필드는 반드시 배열 형태여야 하며, 각 요소는 다음 네 개의 key만 사용합니다:
  - "sequence" : 정수, 1부터 시작하는 씬 번호
  - "timestamp" : 예) "00:00-03:00" 형식의 문자열
  - "action" : 해당 구간에서 화면에 보이는 내용, 카메라 움직임, 분위기를 모두 포함하는 설명 (영어)
  - "audio" : 해당 구간에서 들리는 사운드/효과음/음악 관련 설명 (영어)
- "timeline" 안에는 shot_type, camera_movement 등 다른 key를 추가로 만들지 마세요. 필요한 정보는 모두 "action" 텍스트 안에 녹여서 작성합니다.
- "audio" 필드는 반드시 아래 두 개의 key만 사용합니다:
  - "voice_over" : 내레이션/대사/보이스 관련 요약 (영어)
  - "music" : BGM 또는 음악 스타일, 분위기 (영어)
- "audio" 안에 bgm, sfx 같은 다른 key 이름을 만들지 말고, 모든 음악/사운드 정보는:
  - 전반적인 음악/톤 → "music"
  - 내레이션/보이스 → "voice_over"
  로만 정리합니다.
- "camera_work" 섹션은 전체 영상에 공통으로 적용되는 렌즈, 전역적인 카메라 스타일, 효과 정도만 간단히 채우세요.
  - 씬별 카메라 움직임, 샷 타입, 구체적인 화면 설명은 모두 "timeline" 배열의 "action" 텍스트 안에 포함합니다.
- 카메라, 렌즈, 조명 등은 설명이 없을 때는 당신이 장면에 어울리는 값을 "추천"해서 채우고, 정말 결정하기 어려운 경우에만 "none"을 사용하세요.

※ 중요 규칙:
- 카메라 움직임, 샷 타입, 앵글, 화면 묘사 등 모든 카메라 관련 구체 정보는 반드시 timeline[*].action 내부 문장으로만 표현해야 합니다.
camera_work.notes 또는 camera_work.effects 안에는 절대로 구체적인 카메라 움직임(dolly, zoom, pan, tilt), 샷 타입(wide, medium, close-up), 앵글(high-angle, low-angle) 정보를 넣지 마세요.

※ Voice Over 관련 중요 규칙:
- 만약 사용자가 voice_over 또는 대사(말한 문장)를 제공한 경우,
timeline[*].action 안에는 반드시 '말하고 있는 동작'을 포함해야 합니다.

예:
"speaking softly",
"mouth moving naturally while talking",
"subtle talking motion",
"talking while smiling"

voice_over는 단순 내레이션이 아니라,
인물이 직접 말하고 있는 경우라면 반드시 스타일을 반영해 주세요.

즉, voice_over가 있을 경우:
- timeline[*].action 문장 안에 'speaking' 관련 묘사를 추가해야 합니다.
- 해당 컷에서 인물이 말하고 있는 모습이 시각적으로 묘사되도록 작성하세요.

camera_work 섹션에는 아래와 같은 "전역적인 설정"만 포함해야 합니다:
- 전체 영상에 공통으로 사용되는 렌즈 정보 (예: 35mm, 50mm 등)
- 전체 영상에 공통으로 적용되는 색보정/효과 (예: soft bloom, cinematic grading)
- 전체적인 카메라 톤 (예: overall cinematic tone)

모든 장면별 카메라 동작, 샷 구성, 화면 내용은 timeline[*].action 문장 안에 포함하세요.

[조건 2: 누락 데이터 처리]
- 입력 내용에서 찾을 수 없는 정보는 "none"이라고 기입하세요.
- 단, 가능한 경우에는 입력된 키워드와 전체 분위기를 바탕으로 합리적인 값을 추론해 채우려고 노력한 뒤, 정말 정보가 없을 때만 "none"을 사용합니다.
- JSON 작성 후, 하단에 'ComfyUI 사용 json 프롬프트 중 누락 / none 부분'을 마크다운 리스트로 정리하세요.
  - 예: "- character.appearance.eye_color : 눈 색상 정보 없음"

[조건 3: 미드저니 프롬프트 작성]
- 모든 내용은 영문으로 작성합니다.
- 다음 순서를 반드시 지켜서 한 줄 프롬프트를 구성하세요:
  주제(Topic) → 액션(Action) → 배경(Background) → 카메라 움직임(Camera movement) → 스타일(Style) → 구도(Composition)
- 각 요소는 쉼표(,)로 구분합니다.
- 오디오 / 타임라인에서 유추되는 분위기, 리듬감(느린 롱테이크, 빠른 컷 편집 등)은 Style, Camera movement, Mood 표현에 자연스럽게 반영하세요.
- 출력 예시는 다음과 같은 형식입니다(예시는 그대로 복사하지 말고, 상황에 맞게 새로 작성하세요):

  "a smiling Korean woman in her 20s, working on a laptop at a sunlit cafe terrace, soft camera dolly-in with medium shot, cinematic realistic style with warm tones, rule of thirds composition"

[조건 4: 미드저니 누락 확인]
- 미드저니 프롬프트 작성 후, 부족하거나 빠진 요소(예: 카메라 움직임이 모호함, 조명 스타일이 구체적이지 않음 등)를 하단에 리스트로 정리하세요.
  - 예: "- 카메라 움직임이 구체적이지 않음 (어떤 방향으로 이동하는지 불명확)"

[출력 양식 – 반드시 이 순서를 지키세요]

1️⃣ ComfyUI 사용 json 프롬프트
- 아래 [JSON 템플릿]을 기반으로 한 JSON을, 코드 블럭(```json ... ```) 형식으로 출력합니다.

⚠️ ComfyUI 사용 json 프롬프트 중 누락 / none 부분
- JSON 내에서 "none"으로 남은 항목들을 마크다운 리스트로 정리합니다.

2️⃣ 미드저니 사용 프롬프트
- 한 줄짜리 영문 프롬프트로 출력합니다.
- 구성 순서: Topic, Action, Background, Camera movement, Style, Composition (각 요소는 쉼표로 구분)

⚠️ 미드저니 사용 프롬프트 중 누락부분
- 부족하거나 빠진 요소를 리스트로 정리합니다.

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

  "timeline": [
    {
      "sequence": 1,
      "timestamp": "00:00-03:00",
      "action": "___",
      "audio": "___"
    },
    {
      "sequence": 2,
      "timestamp": "03:00-06:00",
      "action": "___",
      "audio": "___"
    },
    {
      "sequence": 3,
      "timestamp": "06:00-08:00",
      "action": "___",
      "audio": "___"
    }
  ],

  "audio": {
    "voice_over": "___",
    "music": "___"
  },

  "aspect_ratio": "___",
  "requirements": "full-size video without letterboxes"
}

[ComfyUI 사용 json 프롬프트 중 누락 / none 부분 예시]

- character.appearance.eye_color : 눈 색상 정보 없음
- character.appearance.scar : 흉터 유무 정보 없음

[미드저니 사용 프롬프트 출력 예시]

Topic, Action, Background, Camera movement, Style, Composition

[미드저니 사용 프롬프트 중 누락부분 예시]

- 카메라 움직임 관련 구체적인 표현 부족
- 조명 스타일 구체 정보 부족
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
    st.markdown("**사용 모델:** `gpt-4.1-mini` (텍스트 전용, 이미지 모드는 아래에서 별도 모델 사용)")

st.markdown("## 1) 기본 정보")

with st.container():
    st.markdown("### 🎬 프로젝트 기본 설정")
    
    c1, c2, c3 = st.columns([1.2, 0.8, 0.8])

    with c1:
        brand = st.text_input("브랜드 / 프로젝트명", value="니코모리", placeholder="예: NICO MORI")

    with c2:
        aspect = st.selectbox("비율", ["16:9", "9:16", "1:1", "21:9"], index=0)

    with c3:
        duration = st.number_input("길이(초)", min_value=3, max_value=60, value=8, step=1)

# 🔹 prompt_name 누락 방지용 (기본 정보 바로 아래에 배치)
prompt_name = st.text_input("프롬프트 이름 (내가 구분용으로 쓸 제목)", value="카페 테라스 작업 씬")

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

generate_btn = st.button("🚀 프롬프트 생성하기 (텍스트 기반)")

# ==============================================================================
# [3] OpenAI 호출 함수 (텍스트 기반)
# ==============================================================================
def ask_openai(prompt: str) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",  # 필요하면 gpt-4.1 / gpt-4.1-mini 등으로 변경 가능
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


# ==============================================================================
# [4] OpenAI 호출 함수 (이미지 기반)
# ==============================================================================
def ask_openai_with_image(image_bytes: bytes) -> str:
    """
    업로드된 이미지를 기반으로, 현재 SYSTEM_INSTRUCTION에 맞는
    ComfyUI JSON + Midjourney 프롬프트를 생성.
    """
    client = OpenAI(api_key=OPENAI_API_KEY)

    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    user_content = [
        {
            "type": "text",
            "text": (
                "다음 이미지를 분석해서, 위에서 설명한 SYSTEM_INSTRUCTION과 JSON 템플릿 형식에 맞춰 "
                "ComfyUI JSON 프롬프트와 Midjourney용 한 줄 프롬프트를 생성해 주세요. "
                "이미지 속 인물, 배경, 조명, 카메라 구도, 분위기를 최대한 정확하게 분석해서 채우고, "
                "timeline과 audio 섹션은 이 이미지에서 자연스럽게 유추되는 약 6~10초 길이의 시퀀스로 구성해 주세요."
            ),
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64_image}"
            },
        },
    ]

    response = client.chat.completions.create(
        # ⚠️ 이미지 지원되는 모델로 교체 필요할 수 있음 (예: gpt-4o, gpt-4.1 등)
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


# ==============================================================================
# [5] 텍스트 기반 생성 로직
# ==============================================================================
if generate_btn:
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY가 설정되지 않았습니다. Secrets에 'openai_api_key'를 등록해 주세요.")
    else:
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

            left, right = st.columns(2)

            with left:
                st.markdown("### 🧩 전체 결과 (Markdown)")
                st.markdown(result_text)

            with right:
                st.markdown("### 🎨 Midjourney 프롬프트 (코드 복사용)")

                text = result_text

                start_markers = [
                    "2️⃣ 미드저니 사용 프롬프트",
                    "### 2️⃣ 미드저니 사용 프롬프트",
                    "미드저니 사용 프롬프트",
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
                    mj = text[start_index:].strip()

                    end_markers = ["⚠️", "###", "1️⃣", "3️⃣"]
                    end_index = len(mj)
                    for end in end_markers:
                        if end in mj:
                            pos = mj.index(end)
                            end_index = min(end_index, pos)

                    mj = mj[:end_index].strip()
                    mj = mj.replace("```", "").strip()

                    lines = mj.splitlines()
                    if len(lines) > 1:
                        first_line = lines[0]
                        if ("프롬프트" in first_line) or ("Prompt" in first_line):
                            mj = "\n".join(lines[1:]).strip()

                    st.code(mj, language="text")

        except Exception as e:
            st.error(f"실행 중 오류가 발생했습니다: {e}")


# ==============================================================================
# [6] 이미지 업로드 → JSON/MJ 생성 기능
# ==============================================================================
st.markdown("---")
st.markdown("## 7) 이미지에서 프롬프트 / JSON 생성하기 (옵션)")

image_file = st.file_uploader(
    "참고용 이미지 업로드 (jpg, jpeg, png)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    help="레퍼런스 이미지 한 장만 업로드해 주세요."
)

generate_from_image_btn = st.button("🖼 이미지로부터 프롬프트 생성하기")

if generate_from_image_btn:
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY가 설정되지 않았습니다. Secrets에 'openai_api_key'를 등록해 주세요.")
    elif image_file is None:
        st.error("먼저 이미지를 업로드해 주세요.")
    else:
        try:
            image_bytes = image_file.read()

            with st.spinner("이미지를 분석하여 JSON + 프롬프트를 생성하는 중입니다..."):
                result_text = ask_openai_with_image(image_bytes)

            st.success("이미지 기반 프롬프트 생성 완료!")

            left, right = st.columns(2)

            with left:
                st.markdown("### 🧩 전체 결과 (이미지 기반 / Markdown)")
                st.markdown(result_text)

            with right:
                st.markdown("### 🎨 Midjourney 프롬프트 (코드 복사용)")

            # MJ 추출 로직 재사용
                text = result_text

                start_markers = [
                    "2️⃣ 미드저니 사용 프롬프트",
                    "### 2️⃣ 미드저니 사용 프롬프트",
                    "미드저니 사용 프롬프트",
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
                    mj = text[start_index:].strip()

                    end_markers = ["⚠️", "###", "1️⃣", "3️⃣"]
                    end_index = len(mj)
                    for end in end_markers:
                        if end in mj:
                            pos = mj.index(end)
                            end_index = min(end_index, pos)

                    mj = mj[:end_index].strip()
                    mj = mj.replace("```", "").strip()

                    lines = mj.splitlines()
                    if len(lines) > 1:
                        first_line = lines[0]
                        if ("프롬프트" in first_line) or ("Prompt" in first_line):
                            mj = "\n".join(lines[1:]).strip()

                    st.code(mj, language="text")

        except Exception as e:
            st.error(f"이미지 기반 생성 중 오류가 발생했습니다: {e}")

# ==============================================================================
# [Footer]
# ==============================================================================
st.markdown(
    """
    <hr>
    <p style='text-align:center; color: gray; font-size: 14px;'>
    © 2025 NICO MORI. All rights reserved.
    </p>
    """,
    unsafe_allow_html=True
)
