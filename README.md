openai/gpt-3.5-turbo 인증키
https://openrouter.ai/models?q=keys


#파일 다시 chunks
cd ..
python .\embedding_chunks.py

#FastAPI 서버 실행하기
cd api_server 
uvicorn main:app --reload
#접속 url
http://127.0.0.1:8000/docs


# React 프론트 프로젝트 실행
cd .\frontend\
npm start

http://localhost:3000
비밀번호를 잃어버렸어. 어떻게 해?



# github 계정
https://github.com/onepmis-redmine/construction-chatbot
onepmis-redmine
onepmis-redmine@naver.com / onepmis123
# git 반영하기
cd Documents\construction-chatbot
git add .
git commit -m "개발 환경 우선 처리 및 예시 출력 형식 개선"
git push origin main


# Render에 FastAPI 백엔드 배포하기
https://render.com 
행정망에서는 더이상 진행이 안됨. 
노트북에 다시 세팅해서 진행해봐야겠음. 정리도 좀 하고



# faq구조최적화ai
https://makersuite.google.com/app/apikey
내구글계정

# Cursor ai
https://www.cursor.com/
내구글계정, onepmis-redmine@gmail.com




🧪 UI 개선 (Material UI, Tailwind 등)
💬 챗봇 형식 변경 (대화 방식)
🔁 GPT 응답 캐싱 등

# uac74uc124 ucc57ubd07 ud504ub85cuc81dud2b8

## ud504ub85cuc81dud2b8 uad6cuc131

- **API uc11cubc84 (Backend)**: FastAPIub97c uc0acuc6a9ud55c RESTful API uc11cubc84
- **ud504ub860ud2b8uc5d4ub4dc (Frontend)**: Reactub97c uc0acuc6a9ud55c uc6f9 uc778ud130ud398uc774uc2a4
- **ubca1ud130 ub370uc774ud130ubca0uc774uc2a4**: ChromaDBub97c uc0acuc6a9ud55c ubca1ud130 uc800uc7a5 ubc0f uac80uc0c9

## uc8fcuc694 uae30ub2a5

1. FAQ ub370uc774ud130 uc5c5ub85cub4dc ubc0f ucc98ub9ac
2. uc774uc0c1uac10uc9c0 uae30ub260 uc9c8ubb38-ub2f5ubcc0 uae30ub2a5
3. uc6d4ubcc4 ub85cuadf8 uad00ub9ac
4. Render uc2acub9bd ubc29uc9c0 uae30ub2a5

## uc124uce58 ubc0f uc2e4ud589

### uc694uad6cuc0acud56d

- Docker ubc0f Docker Compose
- ub610ub294 Python 3.11 uc774uc0c1 ubc0f Node.js 18 uc774uc0c1

### Dockerub97c uc0acuc6a9ud55c uc124uce58

```bash
# ub808ud3ecuc9c0ud1a0ub9ac ubcf5uc81c
git clone https://github.com/username/construction-chatbot.git
cd construction-chatbot

# Docker ucef4ud3ecuc988ub85c uc2e4ud589
docker-compose up -d
```

### uc218ub3d9 uc124uce58

```bash
# API uc11cubc84 uc124uce58
cd api_server
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# ud504ub860ud2b8uc5d4ub4dc uc124uce58 (ubcc4ub3c4 ud130ubbf8ub110uc5d0uc11c)
cd frontend
npm install
npm start
```

## Renderuc5d0 ubc30ud3ecud558uae30

### ubc30ud3ec uc900ube44

1. Render uacc4uc815uc744 uc0dduc131ud558uace0 ub85cuadf8uc778ud569ub2c8ub2e4.
2. `New Web Service`ub97c uc120ud0ddud558uace0 GitHub ub808ud3ecuc9c0ud1a0ub9acub97c uc5f0uacb0ud569ub2c8ub2e4.
3. ub2e4uc74c uc124uc815uc744 uad6cuc131ud569ub2c8ub2e4:
   - Name: construction-chatbot-api
   - Environment: Docker
   - Region: uac00uae4cuc6b4 uc9c0uc5ed uc120ud0dd
   - Branch: main

### ud658uacbd ubcc0uc218 uc124uc815

Render ub300uc2dcubcf4ub4dcuc5d0uc11c ub2e4uc74c ud658uacbd ubcc0uc218ub97c uc124uc815ud569ub2c8ub2e4:

- `OPENROUTER_API_KEY`: OpenRouter API ud0a4
- `RENDER`: true

## uc0acuc6a9ubc95

1. ube0cub77cuc6b0uc800uc5d0uc11c `http://localhost:3000` ub610ub294 ubc30ud3ecub41c URLuc5d0 uc811uc18dud569ub2c8ub2e4.
2. Excel ud30cuc77c(.xlsx) uc5c5ub85cub4dc ubc84ud2bcuc744 ud074ub9adud558uc5ec FAQ ud30cuc77cuc744 uc5c5ub85cub4dcud569ub2c8ub2e4.
3. uc5c5ub85cub4dc ud6c4 "FAQ uad6cuc870ud654 ucc98ub9ac uc2dcuc791" ubc84ud2bcuc744 ud074ub9adud558uc5ec ub370uc774ud130ub97c ucc98ub9acud569ub2c8ub2e4.
4. ucc98ub9acuac00 uc644ub8ccub418uba74 ucc57ubd07uc5d0 uc9c8ubb38uc744 uc785ub825ud560 uc218 uc788uc2b5ub2c8ub2e4.

## Excel ud30cuc77c ud615uc2dd

Excel ud30cuc77cuc740 ub2e4uc74c ud544uc218 uceecub7fcuc744 ud3ecud568ud574uc57c ud569ub2c8ub2e4:

- `uc9c8ubb38`: FAQ uc9c8ubb38
- `ub2f5ubcc0`: FAQ ub2f5ubcc0

## ud2b8ub7ecube14uc288ud305

### ud30cuc77c uc5c5ub85cub4dc uc624ub958

- python-multipart ud328ud0a4uc9c0uac00 uc124uce58ub418uc5b4 uc788ub294uc9c0 ud655uc778ud569ub2c8ub2e4: `pip install python-multipart`

### ubca1ud130 DB ucd08uae30ud654 uc624ub958

- ub85cuadf8 ud30cuc77cuc744 ud655uc778ud558uc5ec uc624ub958 uba54uc2dcuc9c0ub97c ud655uc778ud569ub2c8ub2e4.
- docs ub514ub809ud1a0ub9acuac00 uc874uc7acud558ub294uc9c0 ud655uc778ud569ub2c8ub2e4.

# 건설 챗봇 프로젝트

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
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 프론트엔드 설치 (별도 터미널에서)
cd frontend
npm install
npm start
```

## 개발 명령어

```bash
# FastAPI 서버 실행
cd api_server 
uvicorn main:app --reload
# 접속 URL: http://127.0.0.1:8000/docs

# React 프론트엔드 실행
cd frontend
npm start
# 접속 URL: http://localhost:3000
```

## Render에 배포하기

### 배포 준비

1. Render 계정을 생성하고 로그인합니다.
2. `New Web Service`를 선택하고 GitHub 레포지토리를 연결합니다.
3. 다음 설정을 구성합니다:
   - Name: construction-chatbot-api
   - Environment: Docker
   - Region: 가까운 지역 선택
   - Branch: main

### 환경 변수 설정

Render 대시보드에서 다음 환경 변수를 설정합니다:

- `OPENROUTER_API_KEY`: OpenRouter API 키
- `RENDER`: true

## 사용법

1. 브라우저에서 `http://localhost:3000` 또는 배포된 URL에 접속합니다.
2. Excel 파일(.xlsx) 업로드 버튼을 클릭하여 FAQ 파일을 업로드합니다.
3. 업로드 후 "FAQ 구조화 처리 시작" 버튼을 클릭하여 데이터를 처리합니다.
4. 처리가 완료되면 챗봇에 질문을 입력할 수 있습니다.

## Excel 파일 형식

Excel 파일은 다음 필수 컬럼을 포함해야 합니다:

- `질문`: FAQ 질문
- `답변`: FAQ 답변

## 트러블슈팅

### 파일 업로드 오류

- python-multipart 패키지가 설치되어 있는지 확인합니다: `pip install python-multipart`

### 벡터 DB 초기화 오류

- 로그 파일을 확인하여 오류 메시지를 확인합니다.
- docs 디렉토리가 존재하는지 확인합니다.
