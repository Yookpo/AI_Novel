import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from datasets import load_dataset
import re

# --- 전역 상수 --- 
GEMINI_MODEL_NAME = 'gemini-1.5-flash-latest'  # 사용할 Gemini AI 모델의 이름
HUGGINGFACE_DATASET = "manu/project_gutenberg" # 사용할 Hugging Face 데이터셋의 이름 (프로젝트 구텐베르크)
TITLE_REGEX = re.compile(r"Title:\s*(.+)", re.IGNORECASE)
MIN_BOOK_LENGTH = 70000
MAX_BOOKS_TO_LIST = 50  # 최종적으로 목록에 포함할 최대 소설 개수
PRIORITY_KEYWORDS = [   # 우선적으로 탐색할 유명 소설 제목
    "sherlock holmes", "scarlet", "baskervilles",
    "pride and prejudice", "frankenstein", "moby dick", "dracula",
    "huckleberry finn", "gatsby", "misérables", "miserables",
    "all quiet on the western front", "rebecca", "wizard of oz",
    "alice in wonderland", "peter pan"
]

def initialize_gemini():
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("Google API 키를 찾을 수 없습니다. .env 파일을 확인해주세요.")
    # genai 라이브러리에 API 키를 설정하여 인증
    genai.configure(api_key=google_api_key)
    return genai.GenerativeModel(GEMINI_MODEL_NAME)

# Hugging Face에서 소설을 찾고 정렬
def find_and_sort_books():
    print("Hugging Face 데이터셋 탐색을 시작합니다... (시간이 걸릴 수 있습니다)")
    ds = load_dataset(HUGGINGFACE_DATASET, split="en", streaming=True)
    
    priority_books = {}
    other_books = {}
    found_keywords = set()
    
    # 최대 30,000개의 문서를 스캔
    SEARCH_LIMIT = 50000
    for i, book_data in enumerate(ds.take(SEARCH_LIMIT)):
        if i > 0 and i % 1000 == 0:
            print(f"... {i:,} / {SEARCH_LIMIT:,}개 문서 스캔 중 ...")
            
        # 목표 소설 개수(50개) 를 모두 채우면 탐색 중지
        if len(priority_books) + len(other_books) >= MAX_BOOKS_TO_LIST:
            print("목표량을 모두 채워 탐색을 종료합니다.")
            break

        text = book_data['text']
        if len(text) < MIN_BOOK_LENGTH: continue
        
        match = TITLE_REGEX.search(text)
        if not match: continue
        
        title = match.group(1).strip()
        if not title or title in priority_books or title in other_books: continue
        
        is_priority = False
        normalized_title = title.lower().replace('-', ' ')
        for kw in PRIORITY_KEYWORDS:
            if kw in normalized_title and kw not in found_keywords:
                priority_books[title] = text
                found_keywords.add(kw)
                is_priority = True
                print(f"✅ 유명 소설 발견: {title}")
                break
        
        if not is_priority:
            if len(other_books) < MAX_BOOKS_TO_LIST - len(PRIORITY_KEYWORDS):
                other_books[title] = text
    
    print("탐색 완료!")
    sorted_titles = list(priority_books.keys()) + list(other_books.keys())
    all_books_data = {**priority_books, **other_books}
    return sorted_titles, all_books_data

# AI모델을 이용해 영어 제목 -> 한국어로
def translate_titles(model, sorted_english_titles):
    print("Gemini API로 제목 번역을 시작합니다...")
    prompt = "Translate the following book titles into Korean. Maintain the original order and provide only the translated titles, one per line. Do not add numbers or bullets.\n\n"
    prompt += "\n".join(sorted_english_titles)
    response = model.generate_content(prompt)
    translated_titles = response.text.strip().split('\n')
    print("번역 완료!")
    if len(sorted_english_titles) == len(translated_titles):
        return {kt: et for kt, et in zip(translated_titles, sorted_english_titles)}
    else:
        print("경고: 번역된 제목과 원본 제목의 개수가 다릅니다. 원본 제목을 사용합니다.")
        return {et: et for et in sorted_english_titles}

if __name__ == "__main__":
    # 1. 모델 초기화
    model = initialize_gemini()
    
    # 2. 책 찾기 및 정렬
    sorted_titles, books_data = find_and_sort_books()
    
    # 3. 제목 번역
    korean_map = translate_titles(model, sorted_titles)
    
    # 4. 결과를 JSON 파일로 저장
    with open("books_data.json", "w", encoding="utf-8") as f:
        json.dump(books_data, f, ensure_ascii=False, indent=4)
    print("✅ 'books_data.json' 파일 저장 완료!")
    
    with open("korean_map.json", "w", encoding="utf-8") as f:
        json.dump(korean_map, f, ensure_ascii=False, indent=4)
    print("✅ 'korean_map.json' 파일 저장 완료!")