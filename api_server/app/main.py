from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import pandas as pd
from pathlib import Path
import json

from app.config import (
    CORS_ORIGINS, 
    FRONTEND_BUILD_DIR, 
    VECTOR_DB_PATH,
    MODEL_CACHE_DIR,
    ENHANCED_FAQ_PATH
)
from app.models.faq import Question, FAQResponse
from app.services.faq_service import FAQService
from app.services.vector_db import VectorDBService
from app.services.embedding import EmbeddingService
from app.utils.logging import setup_logger

# 로거 설정
logger = setup_logger(__name__)

# FastAPI 앱 초기화
app = FastAPI(title="건설업 FAQ 챗봇 API")

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서비스 초기화
vector_db_service = VectorDBService(vector_db_path=VECTOR_DB_PATH)
embedding_service = EmbeddingService(cache_dir=MODEL_CACHE_DIR)
faq_service = FAQService(
    vector_db_service=vector_db_service,
    embedding_service=embedding_service,
    enhanced_faq_path=ENHANCED_FAQ_PATH
)

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행되는 이벤트 핸들러"""
    try:
        # 벡터 DB 초기화
        vector_db_service.initialize()
        
        # 임베딩 모델 로드
        embedding_service.load_model()
        
        # FAQ 데이터 로드
        faq_service.load_enhanced_faq()
        
        logger.info("서버가 성공적으로 시작되었습니다.")
    except Exception as e:
        logger.error(f"서버 시작 중 오류 발생: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 실행되는 이벤트 핸들러"""
    try:
        # 임베딩 모델 언로드
        embedding_service.unload_model()
        logger.info("서버가 정상적으로 종료되었습니다.")
    except Exception as e:
        logger.error(f"서버 종료 중 오류 발생: {str(e)}")

@app.post("/api/faq", response_model=FAQResponse)
async def get_faq_answer(question: Question):
    """FAQ 답변을 제공하는 엔드포인트"""
    try:
        return await faq_service.find_faq_match(question.query)
    except Exception as e:
        logger.error(f"FAQ 처리 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=FAQResponse)
async def ask_question(question: Question):
    """질문에 대한 답변을 제공하는 엔드포인트"""
    try:
        result = faq_service.find_faq_match(question.query)
        if result is None:
            return FAQResponse(
                answer="죄송합니다. 해당 질문에 대한 답변을 찾을 수 없습니다.",
                sources=[],
                is_faq=False
            )
        return result
    except Exception as e:
        logger.error(f"질문 처리 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-excel")
async def upload_excel(file: UploadFile = File(...)):
    """Excel 파일을 업로드하고 FAQ 데이터를 업데이트하는 엔드포인트"""
    try:
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Excel 파일만 업로드 가능합니다.")
        
        # 파일 저장
        file_path = Path("docs") / "enhanced_qa_pairs.xlsx"
        file_path.parent.mkdir(exist_ok=True)
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # FAQ 데이터 다시 로드
        if faq_service.load_enhanced_faq():
            return {"message": "FAQ 데이터가 성공적으로 업데이트되었습니다."}
        else:
            raise HTTPException(status_code=500, detail="FAQ 데이터 로드 중 오류가 발생했습니다.")
            
    except Exception as e:
        logger.error(f"파일 업로드 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-excel")
async def process_excel():
    """업로드된 Excel 파일을 처리하고 벡터 DB에 저장하는 엔드포인트"""
    try:
        # FAQ 데이터 로드 및 처리
        if not faq_service.load_enhanced_faq():
            raise HTTPException(status_code=500, detail="FAQ 데이터 로드 중 오류가 발생했습니다.")
        
        if not faq_service.process_faq_data():
            raise HTTPException(status_code=500, detail="FAQ 데이터 처리 중 오류가 발생했습니다.")
        
        return {"message": "FAQ 데이터가 성공적으로 처리되었습니다."}
        
    except Exception as e:
        logger.error(f"Excel 처리 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 정적 파일 서빙 설정
app.mount("/", StaticFiles(directory=str(FRONTEND_BUILD_DIR), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 