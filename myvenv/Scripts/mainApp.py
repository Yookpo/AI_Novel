import os
import re
import json
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# --- 1. 상수 및 전역 설정 ---
BOOKS_DATA_FILE = "books_data.json" # 소설 원문 데이터 파일 
KOREAN_MAP_FILE = "korean_map.json" # 한글-영문 제목 매핑 파일
GEMINI_MODEL_NAME = 'gemini-1.5-flash-latest'   # 사용할 AI 모델

# --- 2. 핵심 기능 함수 (모델 초기화, 데이터 로딩) ---
def initialize_gemini():
    # API 키를 로드하고 Gemini 모델을 초기화
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    
    if not google_api_key:
        st.error("Google AI API 키를 찾을 수 없습니다. .env 또는 Streamlit secrets에 키를 설정해주세요.")
        st.stop()

    try:
        genai.configure(api_key=google_api_key)
        return genai.GenerativeModel(GEMINI_MODEL_NAME)
    except Exception as e:
        st.error(f"Gemini 모델 초기화 중 오류 발생: {e}")
        st.stop()

@st.cache_data  # 함수의 실행 결과를 캐시, 파일 로딩은 한번, 다시 실행하면 저장된 결과를 즉시 반환
def load_data_from_local_files():
    # 로컬 JSON 파일에서 소설 데이터를 즉시 불러옵니다.
    try:
        with open(BOOKS_DATA_FILE, "r", encoding="utf-8") as f:
            books_data = json.load(f)
        with open(KOREAN_MAP_FILE, "r", encoding="utf-8") as f:
            korean_to_english_map = json.load(f)
        return books_data, korean_to_english_map
    except FileNotFoundError:
        st.error(f"데이터 파일({BOOKS_DATA_FILE}, {KOREAN_MAP_FILE})을 찾을 수 없습니다. `preprocess.py`를 먼저 실행해주세요.")
        st.stop()

# --- 3. AI 생성 기능 함수 ---
# 주어진 소설 본문을 요약하고 한국어로 번역합니다.
def generate_summary(model, novel_text):
    
    # st.spinner: 사용자에게 현재 작업이 진행 중임을 알려주는 로딩 애니메이션
    with st.spinner("📖 소설의 핵심 줄거리를 요약하는 중입니다..."):
        try:
            summary_prompt = f"Please provide a detailed summary of the key events from the following novel:\n\n{novel_text}"
            summary_response = model.generate_content(summary_prompt)
            st.session_state.base_summary = summary_response.text
            
            with st.spinner("🇰🇷 요약된 줄거리를 한국어로 번역하는 중입니다..."):
                translate_prompt = f"Translate the following English text into Korean:\n\n{st.session_state.base_summary}"
                translate_response = model.generate_content(translate_prompt)
                st.session_state.translated_summary = translate_response.text
        except Exception as e:
            st.error(f"요약 생성 중 오류가 발생했습니다: {e}")

def generate_persona_analysis_with_big5(model, character_name, big5_profile, base_summary):
    """Big5 프로필을 기반으로 페르소나 관점의 이야기를 생성합니다."""
    with st.spinner(f"👤 '{character_name}'의 시선으로 소설을 재구성하는 중..."):
        persona_prompt = f"""
You are the character '{character_name}'.
Your personality is defined by the Big 5 model (OCEAN). Based on the provided plot summary, recount the story as if you personally experienced these events. Your narrative should be a first-person account that strongly reflects your unique personality profile through your inner thoughts, feelings, and reactions.
The story must be written in Korean.

---
**Character's Big 5 Personality Profile:**
{big5_profile}
---

**Original Plot Summary:**
{base_summary}
"""
        try:
            persona_response = model.generate_content(persona_prompt)
            st.session_state.perspective_text = persona_response.text
        except Exception as e:
            st.error(f"😭 관점 텍스트 생성 중 오류: {e}")

# --- 4. UI 렌더링 함수 ---

# 페이지 기본 설정
def setup_page():
    st.set_page_config(page_title="AI 소설 분석기", page_icon="📚", layout="wide")

# 앱의 상태를 저장하는 st.session_state를 초기화
def initialize_session_state():
    keys_to_init = ["novel_text", "base_summary", "translated_summary", "perspective_text"]
    for key in keys_to_init:
        if key not in st.session_state:
            st.session_state[key] = ""

# 왼쪽 컬럼에 소설 출처를 선택하는 UI를 렌더링합니다.
def display_source_selection(col, books_data, korean_to_english_map):
    
    with col:
        st.subheader("1. 분석할 소설 선택")
        # st.radio: 여러 옵션 중 하나를 선택하는 라디오 버튼
        source_option = st.radio("소설 출처", ('엄선된 유명 소설', '내 파일 업로드'), horizontal=True, label_visibility="collapsed")

        if source_option == '엄선된 유명 소설':
            korean_titles = list(korean_to_english_map.keys())
            # st.selectbox: 드롭다운 메뉴
            selected_korean_title = st.selectbox("분석할 소설을 선택하세요.", korean_titles, index=None, placeholder="목록에서 소설을 선택해주세요...")
            if selected_korean_title:
                selected_english_title = korean_to_english_map[selected_korean_title]
                st.session_state.novel_text = books_data.get(selected_english_title, "")
        else:
            uploaded_file = st.file_uploader("분석할 소설 텍스트 파일(.txt)을 업로드하세요.", type="txt")
            if uploaded_file:
                try:
                    st.session_state.novel_text = uploaded_file.read().decode('utf-8')
                    st.success(f"✅ '{uploaded_file.name}' 파일이 성공적으로 업로드되었습니다.")
                except Exception as e:
                    st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
                    st.session_state.novel_text = ""

# 왼쪽 컬럼 하단에 요약 및 번역 결과를 렌더링
def display_summary_results(col):
    if st.session_state.base_summary:
        with col:
            st.divider()
            st.subheader("📝 원본 줄거리 요약")
            with st.expander("🇰🇷 한글 번역본 보기"):
                st.markdown(st.session_state.translated_summary)
            with st.expander("🇬🇧 영문 원본 보기"):
                st.markdown(st.session_state.base_summary)

# 오른쪽 컬럼에 Big5 기반 페르소나 입력 폼과 분석 결과를 렌더링
def display_persona_form_and_results(col, model):
    
    with col:
        st.subheader("2. 페르소나 설정 및 분석")
        
        if not st.session_state.base_summary:
            st.info("왼쪽에서 먼저 '줄거리 요약하기'를 실행해주세요.")
            return

        # st.form: 여러 입력 위젯들을 그룹화
        with st.form("persona_form"):
            st.markdown("##### STEP 1. 등장인물 이름")
            character_name = st.text_input("등장인물의 이름", placeholder="예: Sherlock Holmes, 홍길동", label_visibility="collapsed")
            
            st.markdown("##### STEP 2. Big 5 성격 모델 설정")
            c1, c2 = st.columns(2)
            with c1:
                openness = st.slider("🟢 개방성", 0, 100, 50, help="상상력, 호기심, 창의성")
                conscientiousness = st.slider("🔵 성실성", 0, 100, 50, help="책임감, 계획성, 신중함")
                extraversion = st.slider("🟡 외향성", 0, 100, 50, help="사교성, 활동성, 에너지 수준")
            with c2:
                agreeableness = st.slider("🟣 우호성", 0, 100, 50, help="공감 능력, 협조성, 친절함")
                neuroticism = st.slider("🔴 신경성", 0, 100, 50, help="불안, 우울 등 부정적 감정 경향")

            
            submitted = st.form_submit_button("🎭 페르소나 관점으로 재해석하기")

            if submitted:
                if not character_name:
                    st.warning("⚠️ 등장인물의 이름을 입력해주세요.")
                else:
                    def get_desc(score):
                        return "매우 높음" if score > 80 else "높음" if score > 60 else "보통" if score > 40 else "낮음" if score > 20 else "매우 낮음"
                    
                    big5_profile = (
                        f"- **개방성:** {get_desc(openness)} ({openness}/100)\n"
                        f"- **성실성:** {get_desc(conscientiousness)} ({conscientiousness}/100)\n"
                        f"- **외향성:** {get_desc(extraversion)} ({extraversion}/100)\n"
                        f"- **우호성:** {get_desc(agreeableness)} ({agreeableness}/100)\n"
                        f"- **신경성(부정적 정서):** {get_desc(neuroticism)} ({neuroticism}/100)"
                    )
                    
                    generate_persona_analysis_with_big5(model, character_name, big5_profile, st.session_state.base_summary)
        
        # --- 결과 표시 ---
        if st.session_state.perspective_text:
            st.divider()
            st.subheader(f"📖 {character_name}의 시선으로 다시 읽는 소설")
            st.markdown(st.session_state.perspective_text)
        
        
# --- 5. 메인 실행 함수 ---
def main():
    setup_page() # 페이지 기본 설정
    initialize_session_state()  # 세션 상태 초기화
    model = initialize_gemini() # AI 모델 준비
    books_data, korean_to_english_map = load_data_from_local_files()    # 데이터 로딩(json)

    st.title("📚 AI 소설 분석기")   # 앱의 대제목
    st.markdown("---")
    
     # st.columns: 화면을 지정된 비율로 나누어 컬럼 생성 (왼쪽 45%, 오른쪽 55%)
    col1, col2 = st.columns([0.45, 0.55])

    # 왼쪽 컬럼 UI
    with col1:
        display_source_selection(col1, books_data, korean_to_english_map)
        if st.button("📖 1. 줄거리 요약하기", type="primary", use_container_width=True, disabled=not st.session_state.novel_text):
            for key in ["base_summary", "translated_summary", "perspective_text"]:
                st.session_state[key] = ""
            generate_summary(model, st.session_state.novel_text)
        display_summary_results(col1)
    
    # 오른쪽 컬럼 UI
    with col2:
        display_persona_form_and_results(col2, model)

if __name__ == "__main__":
    main()