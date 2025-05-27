import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
from typing import Optional, Dict, List, Union
import asyncio
import json
from datetime import datetime
from pydantic import BaseModel
import pandas as pd
import uuid  # UUID 생성을 위해 추가

# 현재 디렉토리의 모듈 직접 참조
import vector_db
from faq_processing.faq_processor import FAQProcessor
from utils.system_monitor import SystemMonitor
from utils.session_manager import SessionManager
from openrouter_client import OpenRouterClient
from utils.logger import get_logger
from utils.file_downloader import FileDownloader
from utils.file_uploader import FileUploader

# 로거 설정
logger = get_logger(__name__)

# 기본 디렉토리 설정
BASE_DIR = Path(__file__).parent.parent
VECTOR_DB_DIR = BASE_DIR / "vector_db"
DOCS_DIR = BASE_DIR / "docs"
UPLOAD_DIR = BASE_DIR / "uploads"
SESSIONS_DIR = BASE_DIR / "sessions"
LOGS_DIR = BASE_DIR / "logs"

# 디렉토리 생성
for dir_path in [VECTOR_DB_DIR, DOCS_DIR, UPLOAD_DIR, SESSIONS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# FastAPI 앱 생성
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 컴포넌트 초기화
vector_db = vector_db.VectorDB(VECTOR_DB_DIR)
faq_processor = FAQProcessor(DOCS_DIR, vector_db)
session_manager = SessionManager(SESSIONS_DIR)
system_monitor = SystemMonitor()
file_uploader = FileUploader(UPLOAD_DIR, DOCS_DIR)
openrouter_client = OpenRouterClient()
file_downloader = FileDownloader(BASE_DIR)

class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행되는 이벤트 핸들러"""
    try:
        # FAQ 데이터 로드
        if not faq_processor.load_enhanced_faq():
            logger.error("FAQ 데이터 로드 실패")
            raise Exception("FAQ 데이터를 로드할 수 없습니다.")
        
        # 벡터 DB 초기화
        success = await vector_db.initialize_vector_db(faq_processor.faq_data)
        if not success:
            logger.error("벡터 DB 초기화 실패")
            raise Exception("벡터 DB를 초기화할 수 없습니다.")
        
        # 시스템 모니터링 시작
        system_monitor.start_monitoring()
        
        logger.info("서버 시작 완료")
    except Exception as e:
        logger.error(f"서버 시작 중 오류 발생: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 실행되는 이벤트 핸들러"""
    try:
        # 시스템 모니터링 중지
        await system_monitor.stop_monitoring()
        
        # 벡터 DB 정리
        vector_db.unload_model()
        
        logger.info("서버 종료 완료")
    except Exception as e:
        logger.error(f"서버 종료 중 오류 발생: {e}")

@app.post("/ask")
async def ask_question(request: Request) -> Dict:
    """질문에 대한 답변을 생성합니다."""
    try:
        # 요청 데이터 파싱
        content_type = request.headers.get('content-type', '')
        logger.info(f"Content-Type: {content_type}")
        
        try:
            # 요청 본문을 먼저 읽어서 로깅
            body = await request.body()
            logger.info(f"Raw request body: {body}")
            
            if 'application/json' in content_type:
                data = await request.json()
                logger.info(f"JSON 데이터: {data}")
                # 다양한 필드명 시도
                question = data.get('query') or data.get('question') or data.get('text') or data.get('message') or data.get('content')
                session_id = data.get('session_id')
            else:
                form_data = await request.form()
                logger.info(f"Form 데이터: {form_data}")
                # 다양한 필드명 시도
                question = form_data.get('query') or form_data.get('question') or form_data.get('text') or form_data.get('message') or form_data.get('content')
                session_id = form_data.get('session_id')
        except Exception as e:
            logger.error(f"요청 데이터 파싱 중 오류: {e}")
            raise HTTPException(status_code=400, detail="잘못된 요청 형식입니다.")
        
        if not question:
            logger.error("질문이 없음")
            raise HTTPException(status_code=400, detail="질문이 필요합니다. 'query', 'question', 'text', 'message', 또는 'content' 필드 중 하나를 포함해야 합니다.")
        
        logger.info(f"받은 요청 - 질문: {question}, 세션ID: {session_id}")
        
        # 세션 관리
        if not session_id:
            session_id = str(uuid.uuid4())  # UUID로 세션 ID 생성
            session_data = session_manager.create_session(session_id)
            logger.info(f"새 세션 생성: {session_id}")
        else:
            session_data = session_manager.get_session(session_id)
            if not session_data:
                session_data = session_manager.create_session(session_id)
                logger.info(f"세션 재생성: {session_id}")
        
        # 세션 데이터가 딕셔너리가 아니면 빈 딕셔너리로 초기화
        if not isinstance(session_data, dict):
            session_data = {}
            logger.warning(f"세션 데이터가 딕셔너리가 아님: {session_id}")
        
        # 세션 데이터 구조 확인 및 초기화
        if "conversation_history" not in session_data:
            session_data["conversation_history"] = []
            logger.info(f"대화 기록 초기화: {session_id}")
        
        # FAQ 매칭 시도
        logger.info(f"FAQ 매칭 시도: {question}")
        faq_match, similarity = faq_processor.find_faq_match(question)
        if faq_match is not None and not faq_match.empty:
            logger.info(f"FAQ 매칭 성공 (유사도: {similarity})")
            # structured_answer 필드에서 답변 추출
            try:
                if isinstance(faq_match, pd.Series):
                    structured_answer = faq_match.get('structured_answer')
                    question_variations = json.loads(faq_match.get('question_variations', '[]'))
                    matched_question = question_variations[0] if question_variations else None
                    logger.info(f"원본 structured_answer: {structured_answer}")
                    
                    if isinstance(structured_answer, str):
                        try:
                            answer_data = json.loads(structured_answer)
                            logger.info(f"파싱된 answer_data: {answer_data}")
                            response = answer_data  # JSON 객체 그대로 반환
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON 파싱 오류: {e}")
                            response = {"basic_rules": [structured_answer]}
                    else:
                        response = {"basic_rules": [str(structured_answer)]}
                else:
                    response = {"basic_rules": [str(faq_match)]}
                
                logger.info(f"FAQ 응답 변환: {response}")
            except Exception as e:
                logger.error(f"FAQ 응답 변환 중 오류: {e}")
                response = {"basic_rules": [str(faq_match)]}
        else:
            logger.info("FAQ 매칭 실패, OpenRouter API 호출")
            # OpenRouter API 호출
            conversation_history = session_data.get("conversation_history", [])
            if not isinstance(conversation_history, list):
                conversation_history = []
                logger.warning(f"대화 기록이 리스트가 아님: {session_id}")
            
            response = openrouter_client.call_openrouter(
                question,
                conversation_history
            )
        
        # 세션 업데이트
        message = {
            "role": "user",
            "content": question,
            "timestamp": datetime.now().isoformat()
        }
        updated_session = session_manager.update_session(session_id, message)
        if updated_session is None:
            logger.warning(f"세션 업데이트 실패: {session_id}")
            updated_session = session_data
        
        response_message = {
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        }
        updated_session = session_manager.update_session(session_id, response_message)
        if updated_session is None:
            logger.warning(f"세션 업데이트 실패: {session_id}")
            updated_session = session_data
        
        logger.info(f"응답 생성 완료: {response}")
        return {
            "session_id": session_id,
            "response": response,
            "structured_answer": response,
            "is_faq": faq_match is not None and not faq_match.empty,
            "matched_question": matched_question if faq_match is not None and not faq_match.empty else None
        }
    except HTTPException as he:
        logger.error(f"HTTP 예외 발생: {he}")
        raise he
    except Exception as e:
        logger.error(f"질문 처리 중 오류 발생: {e}")
        logger.error(f"상세 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/session/cleanup")
async def cleanup_sessions() -> Dict:
    """만료된 세션을 정리합니다."""
    try:
        count = session_manager.cleanup_expired_sessions()
        return {"message": f"{count}개의 만료된 세션이 정리되었습니다."}
    except Exception as e:
        logger.error(f"세션 정리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check() -> Dict:
    """서버 상태를 확인합니다."""
    try:
        system_info = system_monitor.collect_system_info()
        return {
            "status": "healthy",
            "system_info": system_info
        }
    except Exception as e:
        logger.error(f"상태 확인 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ping")
async def ping() -> Dict:
    """서버 연결을 확인합니다."""
    return {"message": "pong"}

@app.post("/init-db")
async def initialize_database() -> Dict:
    """벡터 데이터베이스를 초기화합니다."""
    try:
        success = await vector_db.initialize_vector_db()
        if success:
            return {"message": "벡터 데이터베이스가 성공적으로 초기화되었습니다."}
        else:
            raise HTTPException(status_code=500, detail="벡터 데이터베이스 초기화 실패")
    except Exception as e:
        logger.error(f"벡터 DB 초기화 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-excel")
async def upload_excel(file: UploadFile = File(...)) -> Dict:
    """Excel 파일을 업로드하고 처리합니다."""
    try:
        # 파일 저장
        success, result = await file_uploader.save_upload_file(file)
        if not success:
            raise HTTPException(status_code=400, detail=result)
        
        # 파일 처리
        file_path = Path(result)
        if not file_uploader.is_file_valid(file_path):
            raise HTTPException(status_code=400, detail="유효하지 않은 파일입니다.")
        
        return {
            "message": "파일이 성공적으로 업로드되었습니다.",
            "file_path": str(file_path)
        }
    except Exception as e:
        logger.error(f"파일 처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-faq")
async def process_faq() -> Dict:
    """업로드된 FAQ 파일을 처리하고 구조화합니다."""
    try:
        # FAQ 처리 로직 실행
        success = await faq_processor.process_faq()
        if success:
            return {
                "message": "FAQ 데이터가 성공적으로 처리되었습니다.",
                "success": True
            }
        else:
            raise HTTPException(status_code=500, detail="FAQ 처리 중 오류가 발생했습니다.")
    except Exception as e:
        logger.error(f"FAQ 처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download-questions")
async def download_questions():
    """저장된 질문 데이터를 CSV 파일로 다운로드합니다."""
    return file_downloader.download_questions()

@app.get("/download-enhanced-qa")
async def download_enhanced_qa():
    """구조화된 FAQ 파일을 다운로드합니다."""
    try:
        return file_downloader.download_enhanced_qa()
    except Exception as e:
        logger.error(f"파일 다운로드 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}")
async def get_session(session_id: str) -> Dict:
    """세션 데이터를 조회합니다."""
    try:
        session_data = session_manager.get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        return session_data
    except Exception as e:
        logger.error(f"세션 조회 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)