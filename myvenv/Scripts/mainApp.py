import os
import re
import json
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# 시각화 및 데이터 처리를 위한 라이브러리
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

# --- 1. 상수 및 전역 설정 ---
BOOKS_DATA_FILE = "books_data.json" #
KOREAN_MAP_FILE = "korean_map.json" #
GEMINI_MODEL_NAME = 'gemini-1.5-flash-latest' #

# --- 2. 핵심 기능 함수 (모델 초기화, 데이터 로딩) ---
def initialize_gemini(): #
    load_dotenv() #
    google_api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY") #
    if not google_api_key: #
        st.error("Google AI API 키를 찾을 수 없습니다. .env 또는 Streamlit secrets에 키를 설정해주세요.") #
        st.stop() #
    try:
        genai.configure(api_key=google_api_key) #
        return genai.GenerativeModel(GEMINI_MODEL_NAME) #
    except Exception as e: #
        st.error(f"Gemini 모델 초기화 중 오류 발생: {e}") #
        st.stop() #

@st.cache_data #
def load_data_from_local_files(): #
    try:
        with open(BOOKS_DATA_FILE, "r", encoding="utf-8") as f: #
            books_data = json.load(f) #
        with open(KOREAN_MAP_FILE, "r", encoding="utf-8") as f: #
            korean_to_english_map = json.load(f) #
        return books_data, korean_to_english_map #
    except FileNotFoundError: #
        st.error(f"데이터 파일({BOOKS_DATA_FILE}, {KOREAN_MAP_FILE})을 찾을 수 없습니다. `preprocess.py`를 먼저 실행해주세요.") #
        st.stop() #

# --- 3. AI 생성 기능 함수 ---
def generate_summary(model, novel_text): #
    with st.spinner("📖 소설의 핵심 줄거리를 요약하는 중입니다..."): #
        try:
            summary_prompt = f"Please provide a detailed summary of the key events from the following novel:\n\n{novel_text}" #
            summary_response = model.generate_content(summary_prompt) #
            st.session_state.base_summary = summary_response.text #
            with st.spinner("🇰🇷 요약된 줄거리를 한국어로 번역하는 중입니다..."): #
                translate_prompt = f"Translate the following English text into Korean:\n\n{st.session_state.base_summary}" #
                translate_response = model.generate_content(translate_prompt) #
                st.session_state.translated_summary = translate_response.text #
        except Exception as e: #
            st.error(f"요약 생성 중 오류가 발생했습니다: {e}") #

def extract_characters_from_summary(model, base_summary):
    with st.spinner("🕵️ 소설 속 주요 등장인물을 찾는 중..."):
        extract_prompt = f"""
        Based on the following novel summary, identify the main characters.
        Your response MUST be a single, valid JSON array containing strings of the character names.
        For example: ["Character A", "Character B", "Character C"]
        Do not include any other text or explanations outside of the JSON array.
        ---
        **Novel Summary:**
        {base_summary}
        """
        try:
            response = model.generate_content(extract_prompt)
            match = re.search(r'\[.*\]', response.text)
            if match:
                st.session_state.character_list = json.loads(match.group(0))
            else:
                st.session_state.character_list = []
                st.warning("AI가 등장인물 목록을 추출하지 못했습니다. 직접 입력해주세요.")
        except Exception as e:
            st.error(f"등장인물 추출 중 오류 발생: {e}")
            st.session_state.character_list = []

def generate_personality_analysis(model, character_name, base_summary):
    with st.spinner(f"🤖 AI가 '{character_name}'의 성격을 분석하는 중입니다..."):
        analysis_prompt = f"""
        Analyze the personality of the character '{character_name}' based on the provided novel summary.
        Your response MUST be a single, valid JSON object and nothing else. Do not include ```json or any other text outside the JSON object.
        The JSON object must contain the following keys:
        - "openness": An integer between 0 and 100.
        - "conscientiousness": An integer between 0 and 100.
        - "extraversion": An integer between 0 and 100.
        - "agreeableness": An integer between 0 and 100.
        - "neuroticism": An integer between 0 and 100.
        - "reasoning": A short string in Korean, explaining the basis for your analysis.
        ---
        **Novel Summary:**
        {base_summary}
        """
        try:
            response = model.generate_content(analysis_prompt)
            cleaned_text = re.sub(r"```json\n?|```", "", response.text.strip())
            return json.loads(cleaned_text)
        except Exception as e:
            st.error(f"성격 분석 중 오류가 발생했습니다: {e}")
            st.error(f"AI 응답 원문: {response.text}")
            return None

def generate_persona_analysis_with_big5(model, character_name, big5_profile, base_summary): #
    with st.spinner(f"👤 '{character_name}'의 시선으로 소설을 재구성하는 중..."): #
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
        """ #
        try:
            persona_response = model.generate_content(persona_prompt) #
            st.session_state.perspective_text = persona_response.text #
        except Exception as e: #
            st.error(f"😭 관점 텍스트 생성 중 오류: {e}") #

def summarize_persona_narrative(model, perspective_text):
    with st.spinner("✍️ 변경된 관점으로 새로운 줄거리를 생성하는 중..."):
        final_summary_prompt = f"""
        The following text is a first-person narrative from a character whose personality was modified.
        Please summarize this story from a third-person perspective.
        Your summary should highlight how the character's modified personality might have led to different actions, feelings, or outcomes in the key events.
        Please write the summary in Korean.
        ---
        **First-person Narrative:**
        {perspective_text}
        """
        try:
            response = model.generate_content(final_summary_prompt)
            st.session_state.final_summary = response.text
        except Exception as e:
            st.error(f"최종 줄거리 생성 중 오류가 발생했습니다: {e}")

# --- 4. UI 렌더링 함수 ---
def create_radar_chart(scores):
    font_path = 'NanumGothic.ttf'
    if not os.path.exists(font_path):
        st.error(f"'{font_path}' 폰트 파일을 찾을 수 없습니다. 폰트를 프로젝트 폴더에 추가해주세요.")
        return None
    font_prop = fm.FontProperties(fname=font_path)

    labels = ['개방성', '성실성', '외향성', '우호성', '신경성']
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    scores += scores[:1]
    angles += angles[:1]
    
    plt.close('all')
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.fill(angles, scores, color='skyblue', alpha=0.4)
    ax.plot(angles, scores, color='blue', linewidth=2)
    ax.set_yticklabels([])
    ax.set_ylim(0, 100)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontproperties=font_prop)
    for angle, score in zip(angles[:-1], scores[:-1]):
        ax.text(angle, score + 10, str(score), fontproperties=font_prop, horizontalalignment='center', size=12, color='navy')
    plt.rcParams['axes.unicode_minus'] = False
    return fig

def setup_page(): #
    st.set_page_config(page_title="AI 소설 분석기", page_icon="📚", layout="wide") #

def initialize_session_state(): #
    keys_to_init = {
        "novel_text": "", "base_summary": "", "translated_summary": "", 
        "perspective_text": "", "final_summary": "",
        "openness": 50, "conscientiousness": 50, "extraversion": 50, "agreeableness": 50, "neuroticism": 50,
        "analysis_reasoning": "", "radar_chart": None, 
        "character_list": [], "character_name_input": ""
    }
    for key, value in keys_to_init.items():
        if key not in st.session_state:
            st.session_state[key] = value

def display_source_selection(col, books_data, korean_to_english_map): #
    with col: #
        st.subheader("1. 분석할 소설 선택") #
        source_option = st.radio("소설 출처", ('엄선된 유명 소설', '내 파일 업로드'), horizontal=True, label_visibility="collapsed") #
        if source_option == '엄선된 유명 소설': #
            korean_titles = list(korean_to_english_map.keys()) #
            selected_korean_title = st.selectbox("분석할 소설을 선택하세요.", korean_titles, index=None, placeholder="목록에서 소설을 선택해주세요...") #
            if selected_korean_title: #
                selected_english_title = korean_to_english_map[selected_korean_title] #
                st.session_state.novel_text = books_data.get(selected_english_title, "") #
        else: #
            uploaded_file = st.file_uploader("분석할 소설 텍스트 파일(.txt)을 업로드하세요.", type="txt") #
            if uploaded_file: #
                try:
                    st.session_state.novel_text = uploaded_file.read().decode('utf-8') #
                    st.success(f"✅ '{uploaded_file.name}' 파일이 성공적으로 업로드되었습니다.") #
                except Exception as e: #
                    st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}") #
                    st.session_state.novel_text = "" #

def display_summary_results(col): #
    if st.session_state.base_summary: #
        with col: #
            st.divider() #
            st.subheader("📝 원본 줄거리 요약") #
            with st.expander("🇰🇷 한글 번역본 보기"): #
                st.markdown(st.session_state.translated_summary) #
            with st.expander("🇬🇧 영문 원본 보기"): #
                st.markdown(st.session_state.base_summary) #

def display_persona_form_and_results(col, model): #
    with col: #
        st.subheader("2. 페르소나 설정 및 분석") #
        if not st.session_state.base_summary: #
            st.info("왼쪽에서 먼저 '줄거리 요약하기'를 실행해주세요.") #
            return #

        st.markdown("##### STEP 1. 등장인물 선택")
        if st.session_state.character_list:
            st.markdown("AI 추천 등장인물:")
            num_buttons = len(st.session_state.character_list)
            cols = st.columns(min(num_buttons, 4))
            for i, char_name in enumerate(st.session_state.character_list):
                if cols[i % 4].button(char_name, key=f"char_{i}"):
                    st.session_state.character_name_input = char_name
        
        st.text_input("등장인물 이름을 직접 입력하거나 위 버튼을 눌러 선택하세요.", key="character_name_input")
        character_name = st.session_state.character_name_input
        
        st.divider()

        if st.button("🤖 AI로 성격 자동 분석하기", disabled=not character_name, use_container_width=True):
            analysis_data = generate_personality_analysis(model, character_name, st.session_state.base_summary)
            if analysis_data:
                st.session_state.openness = analysis_data.get('openness', 50)
                st.session_state.conscientiousness = analysis_data.get('conscientiousness', 50)
                st.session_state.extraversion = analysis_data.get('extraversion', 50)
                st.session_state.agreeableness = analysis_data.get('agreeableness', 50)
                st.session_state.neuroticism = analysis_data.get('neuroticism', 50)
                scores = [st.session_state.openness, st.session_state.conscientiousness, st.session_state.extraversion, st.session_state.agreeableness, st.session_state.neuroticism]
                st.session_state.radar_chart = create_radar_chart(scores)
                st.session_state.analysis_reasoning = analysis_data.get('reasoning', "분석 이유를 가져오지 못했습니다.")
                st.success(f"'{character_name}'의 성격 분석이 완료되었습니다.")

        if st.session_state.radar_chart:
            st.markdown("#### AI 성격 분석 프로필")
            st.pyplot(st.session_state.radar_chart)
            st.info(f"💡 **AI 분석 요약**: {st.session_state.analysis_reasoning}")
        
        st.markdown("---")
        
        with st.form("persona_form"): #
            st.markdown("##### STEP 2. Big 5 성격 모델 수동 조절") #
            c1, c2 = st.columns(2) #
            with c1: #
                openness = st.slider("🟢 개방성", 0, 100, key="openness") #
                conscientiousness = st.slider("🔵 성실성", 0, 100, key="conscientiousness") #
                extraversion = st.slider("🟡 외향성", 0, 100, key="extraversion") #
            with c2: #
                agreeableness = st.slider("🟣 우호성", 0, 100, key="agreeableness") #
                neuroticism = st.slider("🔴 신경성", 0, 100, key="neuroticism") #
            submitted = st.form_submit_button("🎭 페르소나 관점으로 재해석하기", use_container_width=True) #
            if submitted: #
                if not character_name: #
                    st.warning("⚠️ 등장인물의 이름을 입력/선택해주세요.") #
                else: #
                    st.session_state.perspective_text = "" # 재해석 전 이전 결과 초기화
                    st.session_state.final_summary = "" # 최종 요약도 초기화
                    def get_desc(score): #
                        return "매우 높음" if score > 80 else "높음" if score > 60 else "보통" if score > 40 else "낮음" if score > 20 else "매우 낮음" #
                    big5_profile = (f"- **개방성:** {get_desc(openness)} ({openness}/100)\n" f"- **성실성:** {get_desc(conscientiousness)} ({conscientiousness}/100)\n" f"- **외향성:** {get_desc(extraversion)} ({extraversion}/100)\n" f"- **우호성:** {get_desc(agreeableness)} ({agreeableness}/100)\n" f"- **신경성(부정적 정서):** {get_desc(neuroticism)} ({neuroticism}/100)") #
                    generate_persona_analysis_with_big5(model, character_name, big5_profile, st.session_state.base_summary) #
        
        if st.session_state.perspective_text: #
            st.divider() #
            st.subheader(f"📖 {character_name}의 시선으로 다시 읽는 소설") #
            st.markdown(st.session_state.perspective_text) #
            
            st.markdown("---")
            if st.button("🔄 이 관점으로 새로운 줄거리 생성하기", use_container_width=True, type="primary"):
                summarize_persona_narrative(model, st.session_state.perspective_text)
        
        if st.session_state.final_summary:
            st.divider()
            st.subheader("✍️ AI가 재구성한 최종 줄거리")
            st.markdown(st.session_state.final_summary)

# --- 5. 메인 실행 함수 ---
def main(): #
    setup_page() #
    initialize_session_state() #
    model = initialize_gemini() #
    books_data, korean_to_english_map = load_data_from_local_files() #

    st.title("AI 소설 분석기") #
    st.markdown("---") #
    
    col1, col2 = st.columns([0.45, 0.55]) #

    with col1: #
        display_source_selection(col1, books_data, korean_to_english_map) #
        if st.button("📖 1. 줄거리 요약하기", type="primary", use_container_width=True, disabled=not st.session_state.novel_text): #
            keys_to_reset = [
                "base_summary", "translated_summary", "perspective_text", "final_summary",
                "analysis_reasoning", "radar_chart", "character_list", "character_name_input"
            ]
            for key in keys_to_reset:
                st.session_state[key] = "" if key not in ["radar_chart"] else None
            
            generate_summary(model, st.session_state.novel_text) #
            
            if st.session_state.base_summary:
                extract_characters_from_summary(model, st.session_state.base_summary)
        
        display_summary_results(col1) #
    
    with col2: #
        display_persona_form_and_results(col2, model) #

if __name__ == "__main__": #
    main() #