from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from pydantic import BaseModel
import chromadb
import torch
from transformers import AutoTokenizer, AutoModel
import httpx
import json
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import gc
from pathlib import Path
import pandas as pd
from fastapi.staticfiles import StaticFiles
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import asyncio
import time
from datetime import datetime
from utils.logger import get_logger
from utils.session_manager import SessionManager
import shutil
from fastapi.responses import JSONResponse, FileResponse
import sys
import subprocess
import csv
import io

# 모듈별로 로거 생성
logger = get_logger(__name__)

# 프로젝트 루트 디렉토리 설정
ROOT_DIR = Path(__file__).parent.parent
VECTOR_DB_PATH = ROOT_DIR / "vector_db"
DOCS_DIR = ROOT_DIR / "docs"
ENHANCED_FAQ_PATH = DOCS_DIR / "enhanced_qa_pairs.xlsx"
FRONTEND_BUILD_DIR = ROOT_DIR / "frontend" / "build"
LOGS_DIR = ROOT_DIR / "logs"  # 로그 디렉토리 추가
QUESTIONS_DIR = ROOT_DIR / "questions"  # 질문 저장 디렉토리 추가

# 디렉토리 존재 확인 및 생성
VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)  # 로그 디렉토리 생성
QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)  # 질문 저장 디렉토리 생성

# 질문 저장 파일 경로
QUESTIONS_FILE = QUESTIONS_DIR / "saved_questions.csv"

load_dotenv()  # .env 파일 읽기

# 환경 변수에서 API 키 불러오기
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# 모델 캐시 디렉토리 설정
cache_dir = ROOT_DIR / "model_cache"
cache_dir.mkdir(parents=True, exist_ok=True)

# FastAPI 앱 생성
app = FastAPI()

# CORS 설정 - localhost 우선
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # 로컬 개발 환경 우선
        "https://construction-chatbot-api.onrender.com",
        "*",  # 개발 중에는 모든 출처 허용 (프로덕션에서는 제거)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 개발 환경 감지
def is_development_request(request: Request) -> bool:
    """요청이 로컬 개발 환경에서 온 것인지 확인합니다."""
    origin = request.headers.get("origin", "")
    return origin.startswith("http://localhost:")

# ChromaDB 클라이언트 설정 - 전역 컬렉션 이름 상수 정의
COLLECTION_NAME = "construction_manuals"  # 원래 컬렉션 이름으로 되돌림

# ChromaDB 클라이언트 설정
chroma_client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
collection = None  # 시작 시에는 None으로 설정, 초기화 과정에서 생성

def get_collection():
    """현재 컬렉션을 가져오거나 없으면 생성합니다."""
    global collection
    try:
        if collection is None:
            try:
                # 기존 컬렉션 가져오기 시도
                collection = chroma_client.get_collection(name=COLLECTION_NAME)
                logger.info(f"기존 컬렉션을 가져왔습니다: {COLLECTION_NAME}")
            except Exception as e:
                logger.warning(f"기존 컬렉션을 가져오는데 실패했습니다: {e}")
                # 컬렉션 생성
                collection = chroma_client.create_collection(name=COLLECTION_NAME)
                logger.info(f"새 컬렉션을 생성했습니다: {COLLECTION_NAME}")
        
        # 컬렉션 유효성 검사
        dummy_result = collection.count()
        logger.info(f"컬렉션 항목 수: {dummy_result}")
        return collection
    except Exception as e:
        logger.error(f"컬렉션 가져오기/생성 중 오류: {e}")
        # 마지막 시도: 모든 컬렉션 삭제 후 새로 생성
        try:
            chroma_client.delete_collection(name=COLLECTION_NAME)
            collection = chroma_client.create_collection(name=COLLECTION_NAME)
            logger.info(f"컬렉션을 강제로 재생성했습니다: {COLLECTION_NAME}")
            return collection
        except Exception as e2:
            logger.error(f"컬렉션 강제 재생성 중 오류: {e2}")
            raise e2

# FAQ 데이터 로드
faq_data = None

def load_enhanced_faq():
    """구조화된 FAQ 데이터를 로드합니다."""
    global faq_data
    try:
        if not ENHANCED_FAQ_PATH.exists():
            logger.error(f"Enhanced FAQ file not found at: {ENHANCED_FAQ_PATH}")
            return False
            
        faq_data = pd.read_excel(ENHANCED_FAQ_PATH)
        
        if faq_data.empty:
            logger.error("Enhanced FAQ file is empty")
            return False
        
        # 데이터 컬럼 확인 로깅
        logger.info(f"FAQ 데이터 컬럼: {list(faq_data.columns)}")
        logger.info(f"FAQ 데이터 첫 행: \n{faq_data.iloc[0]}")
        
        # 데이터 형식 검증
        for col in ['question_variations', 'structured_answer', 'keywords']:
            if col not in faq_data.columns:
                logger.error(f"Required column '{col}' not found in FAQ data")
                return False
        
        # JSON 필드 파싱 검증
        for idx, row in faq_data.iterrows():
            try:
                for field in ['question_variations', 'structured_answer', 'keywords']:
                    if isinstance(row[field], str):
                        json.loads(row[field])
                    else:
                        logger.warning(f"Row {idx}: {field} is not a string, converting to string")
                        faq_data.at[idx, field] = json.dumps(row[field], ensure_ascii=False)
            except Exception as e:
                logger.error(f"Error parsing JSON in row {idx}: {e}")
                return False
            
        logger.info(f"Enhanced FAQ data loaded successfully: {len(faq_data)} entries found")
        return True
    except Exception as e:
        logger.error(f"Error loading enhanced FAQ: {e}")
        faq_data = None
        return False

# 시작시 FAQ 데이터 로드
load_enhanced_faq()

# 모델 설정
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
    if model is None:
        model = AutoModel.from_pretrained(
            model_name,
            local_files_only=False
        )
        model.eval()  # 평가 모드로 설정
    return tokenizer, model

def unload_model():
    global tokenizer, model
    del tokenizer
    del model
    tokenizer = None
    model = None
    torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()

# Mean Pooling 함수
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# 텍스트를 임베딩으로 변환하는 함수
def get_embeddings(texts):
    tokenizer, model = load_model()
    
    if isinstance(texts, str):
        texts = [texts]
    
    encoded_input = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    
    # 메모리 정리
    unload_model()
    
    return sentence_embeddings[0].tolist() if len(texts) == 1 else sentence_embeddings.tolist()

# 세션 매니저 초기화
session_manager = SessionManager(storage_dir=str(ROOT_DIR / "sessions"))

# 질문 저장 함수
def save_question(query: str, answer: str = None, similarity: float = None, session_id: str = None):
    """사용자 질문과 답변을 CSV 파일에 저장합니다."""
    try:
        # 파일이 존재하지 않으면 헤더와 함께 생성
        file_exists = QUESTIONS_FILE.exists()
        
        # 현재 시간을 원하는 형식으로 포맷팅 (YYYY-MM-DD AM/PM H:MM)
        now = datetime.now()
        formatted_time = now.strftime("%Y-%m-%d %p %I:%M").replace("AM", "오전").replace("PM", "오후")
        
        # BOM을 사용하여 UTF-8로 저장 (Excel에서 한글 인코딩 문제 해결)
        with open(QUESTIONS_FILE, mode='a', newline='', encoding='utf-8-sig') as file:
            writer = csv.writer(file)
            
            # 파일이 새로 생성된 경우 헤더 추가
            if not file_exists:
                writer.writerow(['timestamp', 'session_id', 'query', 'answer', 'similarity'])
            
            # 질문과 답변 저장
            writer.writerow([
                formatted_time,
                session_id or 'unknown',
                query,
                answer or '',
                f"{similarity:.4f}" if similarity is not None else ''
            ])
        
        logger.info(f"질문과 답변 저장 완료: {query}")
        return True
    except Exception as e:
        logger.error(f"질문 저장 중 오류 발생: {e}")
        import traceback
        logger.error(f"상세 오류: {traceback.format_exc()}")
        return False

# 질문 형식 정의
class Question(BaseModel):
    query: str
    session_id: str = None

# OpenRouter 호출 함수
def call_openrouter(prompt: str, conversation_history=None) -> str:
    headers = {
        "Authorization": OPENROUTER_API_KEY,
        "Content-Type": "application/json"
    }
    
    # API 키가 없는 경우 기본 메시지 반환
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
        return "죄송합니다. 질문에 대한 답변을 찾을 수 없습니다. FAQ에 없는 질문은 현재 처리할 수 없습니다."
    
    messages = []
    
    # 시스템 메시지 추가 (존댓말 지시)
    system_message = {
        "role": "system", 
        "content": "당신은 건설 정보 시스템에 대한 전문 지식을 갖춘 챗봇입니다. 항상 정중하고 공손한 존댓말로 답변해주세요."
    }
    messages.append(system_message)
    
    # 대화 기록이 있으면 추가
    if conversation_history:
        for msg in conversation_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    # 현재 사용자 메시지 추가
    messages.append({"role": "user", "content": prompt})
    
    # 대화 기록이 없는 경우에도 시스템 메시지는 유지
    if len(messages) <= 2:  # 시스템 메시지 + 현재 메시지만 있는 경우
        messages = [system_message, {"role": "user", "content": prompt}]
    
    payload = {
        "model": "openai/gpt-3.5-turbo",
        "messages": messages
    }
    
    try:
        response = httpx.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60.0,
            verify=False  # 테스트 환경에서만 사용
        )
        
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
        data = response.json()
        
        return data["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP 오류: {e.response.status_code} - {e.response.text}")
        return "죄송합니다. 질문에 대한 답변을 찾을 수 없습니다. FAQ에 없는 질문은 현재 처리할 수 없습니다."
    except httpx.RequestError as e:
        logger.error(f"요청 오류: {e}")
        return "죄송합니다. 서버 연결에 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
    except Exception as e:
        logger.error(f"OpenRouter API 호출 중 오류 발생: {e}")
        return "죄송합니다. 질문 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."

def find_faq_match(query: str, threshold: float = 0.6):
    """구조화된 FAQ에서 가장 적절한 답변을 찾습니다."""
    
    if faq_data is None or faq_data.empty:
        logger.error("FAQ data is not loaded")
        return None, 0.0
    
    try:
        logger.info(f"Searching for match to query: {query}")
        logger.info(f"Available FAQ entries: {len(faq_data)}")
        
        # 쿼리 전처리
        query = query.lower().strip()
        if query.endswith('?') or query.endswith('.'): 
            query = query[:-1]
        
        # 쿼리 임베딩 생성
        query_embedding = get_embeddings(query)
        
        # 컬렉션 가져오기
        current_collection = get_collection()
            
        logger.info(f"벡터 검색 시작, 컬렉션 이름: {current_collection.name}")
        
        best_match = None
        highest_similarity = -1
        
        # 각 FAQ 항목과 비교
        for idx, row in faq_data.iterrows():
            try:
                variations = json.loads(row['question_variations']) if isinstance(row['question_variations'], str) else row['question_variations']
                logger.info(f"Row {idx} variations: {variations}")
                
                for q in variations:
                    # 질문 전처리
                    q = q.lower().strip()
                    if q.endswith('?') or q.endswith('.'): 
                        q = q[:-1]
                        
                    q_embedding = get_embeddings(q)
                    similarity = torch.cosine_similarity(
                        torch.tensor(query_embedding),
                        torch.tensor(q_embedding),
                        dim=0
                    ).item()
                    
                    logger.info(f"Comparing '{query}' with '{q}': similarity = {similarity}")
                    
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        best_match = row
                        logger.info(f"New best match found: similarity = {similarity}")
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                continue
        
        logger.info(f"Best match similarity: {highest_similarity} (threshold: {threshold})")
        if highest_similarity >= threshold:
            return best_match, highest_similarity
        return None, highest_similarity
    except Exception as e:
        logger.error(f"Error in find_faq_match: {e}")
        return None, 0.0

@app.post("/ask")
async def ask_question(q: Question, request: Request):
    """질문을 처리하고 답변을 반환합니다."""
    try:
        logger.info(f"Received question: {q.query}, session_id: {q.session_id}")
        
        # 개발 환경 여부 확인
        is_dev = is_development_request(request)
        
        # 세션 처리
        if not q.session_id:
            session = session_manager.create_session()
            q.session_id = session.session_id
            logger.info(f"Created new session: {q.session_id}")
        else:
            # 세션이 존재하는지 확인
            session = session_manager.get_session(q.session_id)
            if not session:
                # 존재하지 않는 세션 ID인 경우 새 세션 생성
                session = session_manager.create_session()
                q.session_id = session.session_id
                logger.info(f"Invalid session ID provided, created new: {q.session_id}")
        
        # 사용자 메시지 저장
        session_manager.add_message(q.session_id, "user", q.query)
        
        # FAQ 매칭 시도
        faq_match, similarity = find_faq_match(q.query)
        if faq_match is not None:
            logger.info(f"FAQ match found for query: {q.query}")
            
            # faq_match 객체의 구조 확인
            logger.info(f"FAQ match columns: {list(faq_match.index)}")
            
            # 응답 데이터 생성
            response_data = {
                "session_id": q.session_id,
                "sources": [faq_match.get("source", "FAQ")],
                "is_dev": is_dev,
                "type": "faq",
                "debug_info": {
                    "source": "faq",
                    "confidence": similarity
                }
            }
            
            # 원본 질문 추가
            if 'original_question' in faq_match:
                response_data["original_question"] = faq_match["original_question"]
            
            # 답변 데이터 추가
            if 'answer' in faq_match:
                # 단순 텍스트 답변
                formatted_answer = faq_match["answer"]
                response_data["answer"] = formatted_answer
                session_manager.add_message(q.session_id, "assistant", formatted_answer)
                
                # 질문과 답변 저장
                save_question(q.query, formatted_answer, similarity, q.session_id)
                
            elif 'structured_answer' in faq_match:
                # 구조화된 답변 - JSON 객체 그대로 전달
                structured_answer = faq_match["structured_answer"]
                if isinstance(structured_answer, str):
                    structured_answer = json.loads(structured_answer)
                
                # 구조화된 데이터 전달
                response_data["structured_answer"] = structured_answer
                
                # 간단한 텍스트 형식으로 저장 (세션 저장용)
                simple_text = "FAQ 답변이 제공되었습니다."
                session_manager.add_message(q.session_id, "assistant", simple_text)
                
                # 구조화된 답변을 문자열로 변환하여 저장
                answer_text = json.dumps(structured_answer, ensure_ascii=False)
                save_question(q.query, answer_text, similarity, q.session_id)
                
            else:
                # 답변 없음
                formatted_answer = "죄송합니다. 이 질문에 대한 답변을 찾을 수 없습니다."
                response_data["answer"] = formatted_answer
                session_manager.add_message(q.session_id, "assistant", formatted_answer)
                
                # 질문과 답변 저장
                save_question(q.query, formatted_answer, similarity, q.session_id)
                
                logger.error(f"FAQ match found but no answer field: {faq_match}")
            
            return response_data
        
        # OpenRouter API 호출
        logger.info(f"No FAQ match found, calling OpenRouter for query: {q.query}")
        answer = call_openrouter(q.query, session_manager.get_messages(q.session_id))
        session_manager.add_message(q.session_id, "assistant", answer)
        
        # 질문과 답변 저장 (OpenRouter의 경우 similarity는 0으로 저장)
        save_question(q.query, answer, 0.0, q.session_id)
        
        return {
            "answer": answer,
            "session_id": q.session_id,
            "sources": [],
            "type": "openrouter",
            "is_dev": is_dev,
            "debug_info": {
                "source": "openrouter",
                "similarity": 0.0
            }
        }
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """세션의 대화 기록을 가져옵니다."""
    messages = session_manager.get_messages(session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"messages": messages}

@app.post("/session/cleanup")
async def cleanup_sessions():
    """오래된 세션을 정리합니다."""
    session_manager.cleanup_old_sessions()
    return {"status": "success", "message": "Sessions cleaned up"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/reload-faq")
async def reload_faq():
    """FAQ 데이터를 재로드합니다."""
    load_enhanced_faq()
    return {"status": "FAQ reloaded"}

# 여기에서 initialize_vector_db 함수 수정
async def initialize_vector_db():
    """구조화된 FAQ 데이터를 벡터 데이터베이스에 임베딩합니다."""
    global collection
    try:
        if not load_enhanced_faq():
            logger.error("구조화된 FAQ 데이터를 로드할 수 없습니다.")
            return {"success": False, "message": "구조화된 FAQ 데이터를 로드할 수 없습니다."}
        
        if faq_data is None or faq_data.empty:
            logger.error("FAQ 데이터가 비어 있습니다.")
            return {"success": False, "message": "FAQ 데이터가 비어 있습니다."}
        
        logger.info("Creating vector database from enhanced FAQ data...")
        
        # 기존 컬렉션 삭제 후 재생성
        try:
            chroma_client.delete_collection(name=COLLECTION_NAME)
            logger.info(f"기존 컬렉션 삭제 완료: {COLLECTION_NAME}")
        except Exception as e:
            logger.warning(f"컬렉션 삭제 중 오류 발생 (무시됨): {e}")
            
        # 새 컬렉션 생성
        collection = chroma_client.create_collection(name=COLLECTION_NAME)
        logger.info(f"새 컬렉션이 성공적으로 생성됨: {COLLECTION_NAME}")
        
        texts = []
        metadatas = []
        ids = []
        
        # 각 FAQ 항목에 대해 임베딩 생성
        total_items = len(faq_data)
        logger.info(f"총 처리할 FAQ 항목: {total_items}개")
        
        for idx, row in faq_data.iterrows():
            # 진행률 표시
            progress_percent = (idx + 1) / total_items * 100
            logger.info(f"FAQ 임베딩 진행률: {progress_percent:.1f}% ({idx + 1}/{total_items})")
            
            # JSON 문자열을 파싱
            question_variations = json.loads(row['question_variations']) if isinstance(row['question_variations'], str) else row['question_variations']
            structured_answer = json.loads(row['structured_answer']) if isinstance(row['structured_answer'], str) else row['structured_answer']
            keywords = json.loads(row['keywords']) if isinstance(row['keywords'], str) else row['keywords']
            original_question = row['original_question']
            
            logger.info(f"Processing FAQ item {idx + 1}/{total_items}: {original_question}")
            
            # 원본 질문 임베딩
            texts.append(original_question)
            metadatas.append({
                "type": "original_question",
                "structured_answer": json.dumps(structured_answer, ensure_ascii=False),
                "keywords": json.dumps(keywords, ensure_ascii=False)
            })
            ids.append(f"orig_{idx}")
            
            # 질문 변형들 임베딩
            for var_idx, q in enumerate(question_variations):
                texts.append(q)
                metadatas.append({
                    "type": "question_variation",
                    "original_question": original_question,
                    "structured_answer": json.dumps(structured_answer, ensure_ascii=False),
                    "keywords": json.dumps(keywords, ensure_ascii=False)
                })
                ids.append(f"q_{idx}_{var_idx}")
            
            # 구조화된 답변의 각 부분도 임베딩
            if 'basic_rules' in structured_answer and structured_answer['basic_rules']:
                for part_idx, rule in enumerate(structured_answer['basic_rules']):
                    text = f"기본 규칙: {rule}"
                    texts.append(text)
                    metadatas.append({
                        "type": "answer_part",
                        "part_type": "basic_rule",
                        "original_question": original_question,
                        "structured_answer": json.dumps(structured_answer, ensure_ascii=False),
                        "keywords": json.dumps(keywords, ensure_ascii=False)
                    })
                    ids.append(f"rule_{idx}_{part_idx}")
            
            if 'examples' in structured_answer and structured_answer['examples']:
                for part_idx, example in enumerate(structured_answer['examples']):
                    if isinstance(example, dict):
                        example_text = json.dumps(example, ensure_ascii=False)
                    else:
                        example_text = str(example)
                    
                    text = f"예시: {example_text}"
                    texts.append(text)
                    metadatas.append({
                        "type": "answer_part",
                        "part_type": "example",
                        "original_question": original_question,
                        "structured_answer": json.dumps(structured_answer, ensure_ascii=False),
                        "keywords": json.dumps(keywords, ensure_ascii=False)
                    })
                    ids.append(f"example_{idx}_{part_idx}")
            
            if 'cautions' in structured_answer and structured_answer['cautions']:
                for part_idx, caution in enumerate(structured_answer['cautions']):
                    text = f"주의사항: {caution}"
                    texts.append(text)
                    metadatas.append({
                        "type": "answer_part",
                        "part_type": "caution",
                        "original_question": original_question,
                        "structured_answer": json.dumps(structured_answer, ensure_ascii=False),
                        "keywords": json.dumps(keywords, ensure_ascii=False)
                    })
                    ids.append(f"caution_{idx}_{part_idx}")
        
        logger.info(f"생성할 임베딩 항목 수: {len(texts)}")
        
        # 임베딩 생성 및 저장
        if texts:
            logger.info("임베딩 생성 시작...")
            
            # 임베딩 생성 시 진행률 표시
            embeddings = []
            total_texts = len(texts)
            for i, text in enumerate(texts):
                logger.info(f"임베딩 생성 진행률: {(i + 1) / total_texts * 100:.1f}% ({i + 1}/{total_texts})")
                embedding = get_embeddings(text)
                embeddings.append(embedding)
            
            logger.info("임베딩 생성 완료! 이제 ChromaDB에 저장합니다...")
            
            # 컬렉션에 추가
            collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"벡터 데이터베이스에 {len(texts)}개 항목 추가 완료")
            
            # 간단한 테스트 쿼리 실행
            test_result = collection.query(
                query_embeddings=[get_embeddings("테스트 쿼리")],
                n_results=1
            )
            logger.info(f"벡터 DB 테스트 쿼리 결과: {test_result}")
            
            return {"success": True, "message": f"{len(texts)}개 항목이 벡터 데이터베이스에 추가되었습니다."}
        else:
            logger.error("임베딩할 텍스트가 없습니다.")
            return {"success": False, "message": "임베딩할 텍스트가 없습니다."}
            
    except Exception as e:
        logger.error(f"벡터 데이터베이스 초기화 중 오류 발생: {str(e)}")
        return {"success": False, "message": f"오류: {str(e)}"}

# 파일 변경 감지를 위한 전역 변수
last_modified_time = None
file_check_interval = 5  # 5초마다 확인

class ExcelFileHandler(FileSystemEventHandler):
    def __init__(self, callback):
        self.callback = callback
        self.last_modified = 0
        
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('qa_pairs.xlsx'):
            # 중복 이벤트 방지를 위한 쿨다운 체크
            current_time = time.time()
            if current_time - self.last_modified > 1:  # 1초 쿨다운
                self.last_modified = current_time
                asyncio.create_task(self.callback())

async def check_file_changes():
    """qa_pairs.xlsx 파일의 변경을 주기적으로 확인합니다."""
    global last_modified_time
    
    while True:
        try:
            if ENHANCED_FAQ_PATH.exists():
                current_mtime = os.path.getmtime(ENHANCED_FAQ_PATH)
                if last_modified_time is None:
                    last_modified_time = current_mtime
                elif current_mtime > last_modified_time:
                    logger.info("qa_pairs.xlsx 파일이 변경되었습니다. 벡터 DB를 업데이트합니다.")
                    last_modified_time = current_mtime
                    await initialize_vector_db()
            await asyncio.sleep(file_check_interval)
        except Exception as e:
            logger.error(f"파일 변경 확인 중 오류 발생: {e}")
            await asyncio.sleep(file_check_interval)

# 마지막으로 정적 파일 서빙 설정하기 전에 추가
# Ping 엔드포인트 - 서버 활성 상태 유지용
@app.get("/ping")
async def ping():
    """서버가 활성 상태임을 확인하는 단순 엔드포인트"""
    return {"status": "alive", "timestamp": datetime.now().isoformat()}

# 자동 핑을 위한 백그라운드 태스크
async def keep_alive():
    """서버가 슬립 상태로 전환되지 않도록 주기적으로 자체 핑을 수행"""
    while True:
        try:
            # 5분마다 자체 ping (Render 무료 플랜 타임아웃은 일반적으로 15분)
            await asyncio.sleep(300)
            async with httpx.AsyncClient() as client:
                # 현재 서버 URL 동적 생성
                host = "localhost"
                port = "8000"
                # 환경 변수 확인
                if os.getenv("RENDER") == "true":
                    # Render 환경에서는 외부 URL 사용
                    response = await client.get("https://construction-chatbot-api.onrender.com/ping")
                else:
                    # 개발 환경에서는 로컬 URL 사용
                    response = await client.get(f"http://{host}:{port}/ping")
                
                logger.info(f"자동 핑 응답: {response.status_code}")
        except Exception as e:
            logger.error(f"자동 핑 에러: {e}")
            continue

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행되는 이벤트 핸들러"""
    logger.info("서버 시작: 벡터 데이터베이스 초기화 시작")
    
    # 초기 벡터 DB 초기화
    await initialize_vector_db()
    
    # 파일 감시 시작
    global last_modified_time
    if ENHANCED_FAQ_PATH.exists():
        last_modified_time = os.path.getmtime(ENHANCED_FAQ_PATH)
    
    # 파일 변경 감지 태스크 시작
    asyncio.create_task(check_file_changes())
    
    # 서버 활성 상태 유지를 위한 자동 핑 태스크 시작
    asyncio.create_task(keep_alive())
    
    logger.info("파일 감시 및 자동 핑 시작됨")

# 수동 초기화용 엔드포인트 (필요시 재초기화 가능)
@app.get("/init-db")
async def manual_initialize_database():
    """벡터 데이터베이스를 수동으로 초기화합니다."""
    success = await initialize_vector_db()
    if success:
        return {"status": "success", "message": "데이터베이스가 성공적으로 초기화되었습니다."}
    else:
        return {"status": "error", "message": "데이터베이스 초기화 중 오류가 발생했습니다."}

@app.post("/upload-excel")
async def upload_excel(file: UploadFile = File(...)):
    """Excel 파일을 업로드하고 벡터 데이터베이스를 업데이트합니다."""
    try:
        # 파일 확장자 확인
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Excel 파일(.xlsx 또는 .xls)만 업로드 가능합니다.")
        
        # 파일 저장 경로
        save_path = DOCS_DIR / "qa_pairs.xlsx"
        
        # 파일 저장
        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Excel 파일이 성공적으로 업로드되었습니다: {save_path}")
        
        # 벡터 DB 초기화 (자동 감지 기능이 이미 이 작업을 처리하지만, 명시적으로 호출)
        await initialize_vector_db()
        
        return JSONResponse(
            content={"message": "Excel 파일이 성공적으로 업로드되었고, 벡터 데이터베이스가 업데이트되었습니다."},
            status_code=200
        )
    except Exception as e:
        logger.error(f"Excel 파일 업로드 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Excel 파일 업로드 실패: {str(e)}")

# process-excel 엔드포인트 수정
@app.post("/process-excel")
async def process_excel():
    """업로드된 FAQ 엑셀 파일을 처리하고 임베딩을 생성합니다."""
    try:
        qa_file_path = DOCS_DIR / "qa_pairs.xlsx"
        if not qa_file_path.exists():
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "처리할 엑셀 파일이 없습니다. 먼저 파일을 업로드해주세요."}
            )
        
        logger.info("FAQ 구조화 처리 시작...")
        
        # qa_allinone.py 실행
        script_path = Path(__file__).parent / "qa_allinone.py"
        process = subprocess.run([sys.executable, str(script_path)], 
                                 capture_output=True, text=True, encoding='utf-8')
        
        if process.returncode != 0:
            logger.error(f"FAQ 구조화 처리 실패: {process.stderr}")
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": f"FAQ 구조화 처리 실패: {process.stderr}"}
            )
        
        logger.info("FAQ 구조화 처리 완료. 이제 벡터 데이터베이스를 초기화합니다.")
        
        # 벡터 데이터베이스 초기화 - 완료될 때까지 기다림
        db_result = await initialize_vector_db()
        
        if not db_result["success"]:
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": f"벡터 데이터베이스 초기화 실패: {db_result['message']}"}
            )
        
        return {"success": True, "message": "FAQ 처리 및 벡터 데이터베이스 초기화가 완료되었습니다."}
        
    except Exception as e:
        logger.error(f"FAQ 처리 중 오류 발생: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"FAQ 처리 중 오류 발생: {str(e)}"}
        )

# 정적 파일 서빙 설정은 모든 API 엔드포인트 정의 후 맨 마지막에 위치
if FRONTEND_BUILD_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_BUILD_DIR), html=True), name="frontend")
    logger.info(f"Frontend static files mounted from {FRONTEND_BUILD_DIR}")
else:
    logger.warning(f"Frontend build directory not found at {FRONTEND_BUILD_DIR}. Static file serving is disabled.")

@app.get("/download-questions")
async def download_questions():
    """저장된 질문을 CSV 파일로 다운로드합니다."""
    try:
        # 기존 파일이 있는지 확인
        if not QUESTIONS_FILE.exists():
            # 파일이 없으면 빈 파일 생성
            with open(QUESTIONS_FILE, mode='w', newline='', encoding='utf-8-sig') as file:
                writer = csv.writer(file)
                writer.writerow(['timestamp', 'session_id', 'query', 'answer', 'similarity'])
        else:
            try:
                # 기존 파일 읽기
                with open(QUESTIONS_FILE, mode='r', encoding='utf-8-sig') as file:
                    content = file.read()
                
                # 파일이 비어있거나 BOM만 있는 경우
                if not content or content == '\ufeff':
                    with open(QUESTIONS_FILE, mode='w', newline='', encoding='utf-8-sig') as file:
                        writer = csv.writer(file)
                        writer.writerow(['timestamp', 'session_id', 'query', 'answer', 'similarity'])
                    logger.info("빈 파일에 헤더를 추가했습니다.")
                else:
                    # 파일 다시 열기
                    with open(QUESTIONS_FILE, mode='r', encoding='utf-8-sig') as file:
                        reader = csv.reader(file)
                        rows = list(reader)
                    
                    # 헤더 확인
                    if len(rows) > 0:
                        header = rows[0]
                        expected_header = ['timestamp', 'session_id', 'query', 'answer', 'similarity']
                        
                        # 헤더가 없거나 예상과 다른 경우
                        if not header or set(header) != set(expected_header):
                            logger.info(f"현재 헤더: {header}")
                            
                            # 헤더 매핑 생성 (기존 헤더 -> 인덱스)
                            header_map = {}
                            if header:
                                for i, h in enumerate(header):
                                    header_map[h.lower()] = i
                            
                            # 새 행 생성
                            new_rows = [expected_header]
                            
                            # 데이터 행 처리
                            for i in range(1, len(rows)):
                                row = rows[i]
                                new_row = [''] * 5  # 5개 열 초기화
                                
                                # 매핑에 따라 데이터 복사
                                if 'timestamp' in header_map and header_map['timestamp'] < len(row):
                                    new_row[0] = row[header_map['timestamp']]
                                elif 0 < len(row):  # 첫 번째 열이 timestamp라고 가정
                                    new_row[0] = row[0]
                                    
                                if 'session_id' in header_map and header_map['session_id'] < len(row):
                                    new_row[1] = row[header_map['session_id']]
                                elif 1 < len(row):  # 두 번째 열이 session_id라고 가정
                                    new_row[1] = row[1]
                                    
                                if 'query' in header_map and header_map['query'] < len(row):
                                    new_row[2] = row[header_map['query']]
                                elif 2 < len(row):  # 세 번째 열이 query라고 가정
                                    new_row[2] = row[2]
                                    
                                if 'answer' in header_map and header_map['answer'] < len(row):
                                    new_row[3] = row[header_map['answer']]
                                elif 3 < len(row):  # 네 번째 열이 answer라고 가정
                                    new_row[3] = row[3]
                                    
                                if 'similarity' in header_map and header_map['similarity'] < len(row):
                                    new_row[4] = row[header_map['similarity']]
                                elif 4 < len(row):  # 다섯 번째 열이 similarity라고 가정
                                    new_row[4] = row[4]
                                
                                # 타임스탬프 형식 변환
                                if new_row[0]:
                                    try:
                                        # ISO 형식 문자열을 datetime 객체로 변환 시도
                                        dt = datetime.fromisoformat(new_row[0].replace('Z', '+00:00'))
                                        # 원하는 형식으로 변환
                                        new_row[0] = dt.strftime("%Y-%m-%d %p %I:%M").replace("AM", "오전").replace("PM", "오후")
                                    except ValueError:
                                        # 변환 실패 시 원본 값 유지
                                        pass
                                
                                new_rows.append(new_row)
                            
                            # 파일 업데이트
                            with open(QUESTIONS_FILE, mode='w', newline='', encoding='utf-8-sig') as file:
                                writer = csv.writer(file)
                                for row in new_rows:
                                    writer.writerow(row)
                            
                            logger.info("CSV 파일 헤더를 업데이트했습니다.")
            except Exception as e:
                logger.error(f"파일 처리 중 오류: {e}")
                import traceback
                logger.error(f"상세 오류: {traceback.format_exc()}")
        
        # 파일 다운로드
        return FileResponse(
            path=QUESTIONS_FILE,
            filename="saved_questions.csv",
            media_type="text/csv"
        )
    except Exception as e:
        logger.error(f"질문 다운로드 중 오류 발생: {e}")
        import traceback
        logger.error(f"상세 오류: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))