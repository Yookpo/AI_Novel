## 실행 방법

1.  **저장소 클론**
    ```bash
    git clone [https://github.com/](https://github.com/)<YOUR_USERNAME>/ai-novel-analyzer.git
    cd ai-novel-analyzer
    ```
2.  **가상 환경 생성 및 활성화**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
3.  **라이브러리 설치**
    ```bash
    pip install -r requirements.txt
    ```
4.  **API 키 설정**
    `.env` 파일을 생성하고 `GOOGLE_API_KEY="YOUR_API_KEY"` 형식으로 Gemini API 키를 추가합니다.

5.  **데이터 전처리**
    ```bash
    python preprocess.py
    ```
6.  **애플리케이션 실행**
    ```bash
    streamlit run mainApp.py
    ```
