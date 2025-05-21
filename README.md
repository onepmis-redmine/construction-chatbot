# 건설 챗봇 프로젝트

## 개발 환경 설정

### API 키 설정
- OpenAI GPT-3.5-turbo: https://openrouter.ai/models?q=keys
- Google AI: https://makersuite.google.com/app/apikey

### 개발 도구
- GitHub 저장소: https://github.com/onepmis-redmine/construction-chatbot
- Cursor AI: https://www.cursor.com/ 계정 BIS-SEOUL(bolim.seoul@gmail.com)


### 배포
-  Render: https://render.com   
-  배포 url: https://construction-chatbot-api.onrender.com



## 프로젝트 구성

- **API 서버 (Backend)**: FastAPI를 사용한 RESTful API 서버
- **프론트엔드 (Frontend)**: React를 사용한 웹 인터페이스
- **벡터 데이터베이스**: ChromaDB를 사용한 벡터 저장 및 검색

## 주요 기능

1. FAQ 데이터 업로드 및 처리
2. 이상감지 기반 질문-답변 기능
3. 월별 로그 관리
4. Render 슬립 방지 기능

## 설치 및 실행

### 요구사항

- Docker 및 Docker Compose
- 또는 Python 3.11 이상 및 Node.js 18 이상

### Docker를 사용한 설치

```bash
# 레포지토리 복제
git clone https://github.com/onepmis-redmine/construction-chatbot.git
cd construction-chatbot

# Docker 컴포즈로 실행
docker-compose up -d
```

### 수동 설치

```bash
# API 서버 설치
cd api_server
# pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 프론트엔드 설치 (별도 터미널에서)
cd frontend
# npm install
npm start
```

## 개발 명령어

### 백엔드 개발
```bash
# FastAPI 서버 실행
cd api_server 
uvicorn main:app --reload
# 접속 URL: http://127.0.0.1:8000/docs

# 데이터 처리
python embedding_chunks.py
```

### 프론트엔드 개발
```bash
# React 프론트엔드 실행
cd frontend
npm start
# 접속 URL: http://localhost:3000
```

### Git 작업
```bash
# 변경사항 반영
git add .
git commit -m "커밋 메시지"
git push origin main
```


🧪 UI 개선 (Material UI, Tailwind 등)
💬 챗봇 형식 변경 (대화 방식)
🔁 GPT 응답 캐싱 등
