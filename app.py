import json
import logging
import os

import openai
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("translator")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

LANGUAGES = {
    "en": ("🇺🇸", "English"),
    "ja": ("🇯🇵", "日本語"),
    "vi": ("🇻🇳", "Tiếng Việt"),
}

# Windows 기본 이모지 글꼴은 국기를 "US" 같은 글자로 표시하므로 국기만 Noto Color Emoji로 그린다.
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Color+Emoji&display=swap');
.flag { font-family: 'Noto Color Emoji', sans-serif; }
/* 마크다운 제목(####)은 자동 앵커 링크가 붙으므로 카드 제목은 div로 그린다. */
.lang-title { font-size: 1.25rem; font-weight: 600; margin-bottom: 0.5rem; }
/* 번역 결과는 복사 버튼 때문에 st.code로 표시하지만, 읽기 쉽게 본문 글꼴을 쓴다. */
[data-testid="stCode"] code {
    font-family: 'Source Sans', sans-serif;
    font-size: 1rem;
    line-height: 1.6;
}
</style>
"""

SYSTEM_PROMPT = """You are a professional translator.
Detect the language of the user's text automatically and translate it into
English (en), Japanese (ja) and Vietnamese (vi).
If the text is already in one of these languages, return a natural, polished
version of the original for that language.
Preserve the meaning, tone, emojis and special characters.
Translate every line of the text completely, line by line:
- Keep exactly the same number of lines and the same line breaks as the original.
- Never merge, split, summarize, shorten or omit lines.
- Even if some lines or sentences are identical, translate each occurrence separately.
Respond with JSON only, exactly in this form:
{"en": "...", "ja": "...", "vi": "..."}"""

MAX_ATTEMPTS = 2  # JSON 형식이 잘못 오면 1회 재시도
MAX_INPUT_CHARS = 2000


class TranslationFormatError(Exception):
    """모델 응답이 올바른 JSON({"en", "ja", "vi"})이 아닐 때 발생한다."""


class TranslationTruncatedError(Exception):
    """응답이 길이 제한으로 잘렸을 때 발생한다. 다시 요청해도 같으므로 재시도하지 않는다."""


@st.cache_resource
def get_client() -> openai.OpenAI:
    return openai.OpenAI(api_key=OPENAI_API_KEY, timeout=30)


def parse_translation(content: str | None) -> dict:
    try:
        data = json.loads(content or "")
    except json.JSONDecodeError as e:
        raise TranslationFormatError("응답이 JSON 형식이 아닙니다.") from e
    if not isinstance(data, dict) or not all(
        isinstance(data.get(code), str) for code in LANGUAGES
    ):
        raise TranslationFormatError("응답에 en/ja/vi 번역이 모두 들어있지 않습니다.")
    return {code: data[code] for code in LANGUAGES}


def translate(text: str) -> dict:
    """입력 텍스트를 영어/일본어/베트남어로 번역해 {"en", "ja", "vi"} 형태로 반환한다."""
    for attempt in range(MAX_ATTEMPTS):
        response = get_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise TranslationTruncatedError
        try:
            return parse_translation(choice.message.content)
        except TranslationFormatError:
            if attempt == MAX_ATTEMPTS - 1:
                raise


def error_message(e: Exception) -> str:
    if isinstance(e, openai.AuthenticationError):
        return "API 키 인증에 실패했습니다. .env 파일의 OPENAI_API_KEY가 올바른지 확인하세요."
    if isinstance(e, openai.RateLimitError):
        return "요청 한도를 초과했습니다. 잠시 후 다시 시도하세요. (사용량/결제 한도도 확인해 보세요)"
    if isinstance(e, openai.APITimeoutError):
        return "응답 시간이 초과되었습니다. 잠시 후 다시 시도하세요."
    if isinstance(e, openai.APIConnectionError):
        return "OpenAI 서버에 연결할 수 없습니다. 인터넷 연결을 확인하세요."
    if isinstance(e, openai.APIError):
        return "OpenAI API 오류가 발생했습니다. 잠시 후 다시 시도하세요."
    if isinstance(e, TranslationTruncatedError):
        return "입력이 너무 길어 번역이 잘렸습니다. 텍스트를 나눠서 번역하세요."
    if isinstance(e, TranslationFormatError):
        return "번역 결과를 읽지 못했습니다. 다시 시도하세요."
    return "알 수 없는 오류가 발생했습니다. 다시 시도하세요."


def log_error(e: Exception):
    # openai 오류 메시지에는 키 일부가 들어갈 수 있으므로 종류와 상태 코드만 남긴다.
    logger.warning(
        "translation failed: %s (status=%s)",
        type(e).__name__,
        getattr(e, "status_code", None),
    )


def count_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def clear():
    st.session_state.source_text = ""
    st.session_state.result = None
    st.session_state.translated_text = None


st.set_page_config(page_title="AI 번역기", page_icon="🌐", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.session_state.setdefault("source_text", "")
st.session_state.setdefault("result", None)
st.session_state.setdefault("translated_text", None)  # result가 어떤 입력의 번역인지
st.session_state.setdefault("auth_failed", False)

st.title("🌐 AI 번역기")
st.caption("입력한 텍스트를 영어 · 일본어 · 베트남어로 번역합니다")

if not OPENAI_API_KEY:
    st.error("API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")

st.text_area(
    "번역할 텍스트",
    key="source_text",
    height=150,
    max_chars=MAX_INPUT_CHARS,
    placeholder="예: 오늘 회의는 3시에 시작합니다.",
)

with st.container(horizontal=True):
    translate_clicked = st.button(
        "번역하기", type="primary", icon=":material/translate:", disabled=not OPENAI_API_KEY
    )
    st.button("지우기", on_click=clear)

# API 호출은 버튼을 누른 rerun에서만 일어난다.
if translate_clicked:
    text = st.session_state.source_text.strip()
    if not text:
        st.warning("번역할 텍스트를 입력하세요.")
    else:
        try:
            with st.spinner("번역 중..."):
                st.session_state.result = translate(text)
            st.session_state.translated_text = text
            st.session_state.auth_failed = False
        except Exception as e:
            log_error(e)
            st.session_state.auth_failed = isinstance(e, openai.AuthenticationError)
            st.error(error_message(e))

# 사이드바는 인증 결과를 반영하도록 번역 처리 뒤에 그린다.
with st.sidebar:
    st.header("설정")
    st.write(f"모델: `{OPENAI_MODEL}`")
    if not OPENAI_API_KEY:
        key_status = "❌ 없음"
    elif st.session_state.auth_failed:
        key_status = "⚠️ 인증 실패"
    else:
        key_status = "✅ 입력됨"
    st.write(f"API 키: {key_status}")

st.divider()

result = st.session_state.result
translated_text = st.session_state.translated_text
if result:
    if translated_text != st.session_state.source_text.strip():
        st.info("아래 결과는 이전 입력의 번역입니다. 지금 입력을 번역하려면 [번역하기]를 누르세요.")
    source_lines = count_lines(translated_text)
    if any(count_lines(result[code]) != source_lines for code in LANGUAGES):
        st.warning("원문과 번역 결과의 줄 수가 다릅니다. 일부 내용이 빠졌을 수 있으니 확인하세요.")

columns = st.columns(3)
for column, (code, (flag, name)) in zip(columns, LANGUAGES.items()):
    with column, st.container(border=True):
        st.markdown(
            f'<div class="lang-title"><span class="flag">{flag}</span> {name}</div>',
            unsafe_allow_html=True,
        )
        if result:
            st.code(result[code], language=None, wrap_lines=True)
        else:
            st.caption("번역 결과가 여기에 표시됩니다")
