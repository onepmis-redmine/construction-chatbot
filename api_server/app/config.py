import os
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트 디렉토리 설정
ROOT_DIR = Path(__file__).parent.parent.parent
VECTOR_DB_PATH = ROOT_DIR / "vector_db"
DOCS_DIR = ROOT_DIR / "docs"
ENHANCED_FAQ_PATH = DOCS_DIR / "enhanced_qa_pairs.xlsx"
FRONTEND_BUILD_DIR = ROOT_DIR / "frontend" / "build"
LOGS_DIR = ROOT_DIR / "logs"
QUESTIONS_DIR = ROOT_DIR / "questions"
QUESTIONS_FILE = QUESTIONS_DIR / "saved_questions.csv"

# 환경 변수 로드
load_dotenv()

# API 키
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# 모델 설정
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE_DIR = ROOT_DIR / "model_cache"

# 벡터 DB 설정
COLLECTION_NAME = "construction_manuals"

# CORS 설정
CORS_ORIGINS = [
    "http://localhost:3000",  # 로컬 개발 환경
    "https://construction-chatbot-api.onrender.com",
    "https://construction-chatbot-frontend.onrender.com",
    "*",  # 개발 중에는 모든 출처 허용 (프로덕션에서는 제거)
]

# 디렉토리 생성
for directory in [VECTOR_DB_PATH, DOCS_DIR, LOGS_DIR, QUESTIONS_DIR, MODEL_CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True) 