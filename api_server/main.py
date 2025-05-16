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
import shutil
from fastapi.responses import JSONResponse
import sys

# 모듈별로 로거 생성
logger = get_logger(__name__)

# 프로젝트 루트 디렉토리 설정
ROOT_DIR = Path(__file__).parent.parent
VECTOR_DB_PATH = ROOT_DIR / "vector_db"
DOCS_DIR = ROOT_DIR / "docs"
ENHANCED_FAQ_PATH = DOCS_DIR / "enhanced_qa_pairs.xlsx"
FRONTEND_BUILD_DIR = ROOT_DIR / "frontend" / "build"
LOGS_DIR = ROOT_DIR / "logs"  # 로그 디렉토리 추가

# 디렉토리 존재 확인 및 생성
VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)  # 로그 디렉토리 생성

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

# ChromaDB 클라이언트 설정
chroma_client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
collection = chroma_client.get_or_create_collection(name="construction_manuals")

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

# 질문 형식 정의
class Question(BaseModel):
    query: str

# OpenRouter 호출 함수
def call_openrouter(prompt: str) -> str:
    headers = {
        "Authorization": OPENROUTER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = httpx.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60.0,
        verify=False  # 테스트 환경에서만 사용
    )

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return f"❌ OpenRouter 응답 오류:\n{json.dumps(data, indent=2, ensure_ascii=False)}"

def find_faq_match(query: str, threshold: float = 0.6):
    """구조화된 FAQ에서 가장 적절한 답변을 찾습니다."""
    if faq_data is None or faq_data.empty:
        logger.error("FAQ data is not loaded")
        return None
    
    try:
        logger.info(f"Searching for match to query: {query}")
        logger.info(f"Available FAQ entries: {len(faq_data)}")
        
        # 쿼리 전처리
        query = query.lower().strip()
        if query.endswith('?') or query.endswith('.'): 
            query = query[:-1]
        
        # 쿼리 임베딩 생성
        query_embedding = get_embeddings(query)
        
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
            return best_match
        return None
    except Exception as e:
        logger.error(f"Error in find_faq_match: {e}")
        return None

@app.post("/ask")
async def ask_question(q: Question, request: Request):
    query = q.query.strip()
    logger.info(f"Received question: {query}")
    
    # 개발 환경 감지
    is_dev = is_development_request(request)
    logger.info(f"Request from development environment: {is_dev}")
    
    try:
        # 벡터 검색 수행
        query_embedding = get_embeddings(query)
        logger.info("Generated query embedding")
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )
        
        logger.info(f"Vector search results: {results}")
        
        # 결과가 비어있는 경우 처리
        if not results["ids"] or len(results["ids"][0]) == 0:
            logger.warning("No results found in vector database")
            return {
                "answer": "죄송합니다. 해당 질문에 대한 답변을 찾을 수 없습니다. 데이터베이스에 데이터가 아직 로드되지 않았을 수 있습니다.",
                "sources": [],
                "is_faq": False,
                "is_dev": is_dev
            }
            
        # 가장 유사한 문서 찾기
        best_match_idx = 0
        best_match_distance = results["distances"][0][best_match_idx]
        
        logger.info(f"Best match distance: {best_match_distance}")
        
        # 개발 환경에서는 더 관대한 임계값 적용
        threshold = 1.8 if is_dev else 1.5
        
        if best_match_distance > threshold:  # 거리가 너무 멀면 관련이 없는 것으로 판단
            logger.warning(f"Best match distance ({best_match_distance}) too high")
            return {
                "answer": "죄송합니다. 해당 질문과 충분히 관련된 답변을 찾지 못했습니다.",
                "sources": [],
                "is_faq": False,
                "is_dev": is_dev
            }
            
        # 메타데이터에서 구조화된 답변 추출
        metadata = results["metadatas"][0][best_match_idx]
        structured_answer = json.loads(metadata["structured_answer"])
        
        # 텍스트 기반 응답 생성
        answer_parts = []
        
        # 메인 답변 섹션
        if structured_answer.get('main'):
            answer_parts.append(f"{structured_answer['main']}\n")
        
        # 기본 규칙 섹션
        if structured_answer.get('basic_rules'):
            # 메인 답변과 중복되지 않도록 처리
            if len(structured_answer['basic_rules']) == 1 and structured_answer['basic_rules'][0] == structured_answer.get('main', ''):
                # main 필드와 basic_rules의 첫 항목이 동일한 경우 중복 표시하지 않음
                pass
            else:
                answer_parts.append("\n[기본 규칙]\n")
                for rule in structured_answer['basic_rules']:
                    # main 필드와 중복되지 않는 경우에만 표시
                    if rule != structured_answer.get('main', ''):
                        answer_parts.append(f"• {rule}\n")
        
        # 추가 정보 섹션
        if structured_answer.get('additional'):
            answer_parts.append(f"\n[추가 정보]\n{structured_answer['additional']}\n")
        
        # 예시 섹션
        if structured_answer.get('examples'):
            answer_parts.append("\n[예시]\n")
            for example in structured_answer['examples']:
                if isinstance(example, dict):
                    # 딕셔너리 형태의 예시를 읽기 쉽게 포맷팅
                    scenario = example.get('scenario', '')
                    
                    # 결과 필드(result로 시작하는 키) 값들 수집
                    result_items = []
                    for k, v in example.items():
                        if k.startswith('result'):
                            result_items.append(f"{v}")
                    
                    # 결과 문자열 생성
                    result_str = ", ".join(result_items) if result_items else ""
                    
                    # 설명 필드가 있는 경우에만 사용
                    explanation = example.get('explanation', '')
                    
                    # 예시 문자열 생성
                    formatted_example = f"• {scenario}"
                    if result_str:
                        formatted_example += f"\n  → {result_str}"
                    if explanation and explanation != result_str:  # 설명이 결과와 다른 경우에만 표시
                        formatted_example += f"\n  ☞ {explanation}"
                    
                    answer_parts.append(f"{formatted_example}\n")
                else:
                    # 일반 문자열 형태의 예시
                    answer_parts.append(f"• {example}\n")
        
        # 주의사항 섹션
        if structured_answer.get('cautions'):
            answer_parts.append("\n[주의사항]\n")
            for caution in structured_answer['cautions']:
                answer_parts.append(f"⚠️ {caution}\n")
        
        # 모든 섹션을 결합하고 불필요한 공백 제거
        answer = "".join(answer_parts).strip()
        
        # 개발 환경에서는 추가 디버그 정보 제공
        debug_info = {}
        if is_dev:
            debug_info = {
                "match_distance": best_match_distance,
                "threshold": threshold,
                "original_query": query,
                "matched_document": results["documents"][0][best_match_idx]
            }
        
        return {
            "answer": answer,
            "sources": ["FAQ 데이터베이스"],
            "is_faq": True,
            "original_question": metadata.get("original_question", "Unknown"),
            "match_distance": best_match_distance,
            "is_dev": is_dev,
            "debug_info": debug_info if is_dev else {}
        }
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return {
            "answer": "죄송합니다. 질문 처리 중 오류가 발생했습니다.",
            "sources": [],
            "is_faq": False,
            "error": str(e) if is_dev else "Internal server error",
            "is_dev": is_dev
        }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/reload-faq")
async def reload_faq():
    """FAQ 데이터를 재로드합니다."""
    load_enhanced_faq()
    return {"status": "FAQ reloaded"}

# 벡터 DB 초기화 함수
async def initialize_vector_db():
    """벡터 데이터베이스를 초기화합니다."""
    try:
        logger.info("벡터 DB 초기화 시작")
        
        # FAQ 데이터 로드
        logger.info(f"FAQ 데이터 파일 경로: {ENHANCED_FAQ_PATH}")
        if not ENHANCED_FAQ_PATH.exists():
            logger.error(f"FAQ 데이터 파일이 존재하지 않음: {ENHANCED_FAQ_PATH}")
            return False
            
        if not load_enhanced_faq():
            logger.error("FAQ 데이터 로드 실패")
            return False
        
        logger.info(f"FAQ 데이터 로드 완료: {len(faq_data) if faq_data is not None else 0}개 항목")
        
        if faq_data is None or len(faq_data) == 0:
            logger.error("FAQ 데이터가 비어있습니다.")
            return False
        
        # 기존 컬렉션 삭제 후 재생성
        try:
            logger.info("기존 컬렉션 삭제 시도")
            chroma_client.delete_collection(name="construction_manuals")
            logger.info("기존 컬렉션 삭제 완료")
        except Exception as e:
            logger.warning(f"기존 컬렉션 삭제 중 예외 발생 (무시 가능): {e}")
        
        logger.info("새 컬렉션 생성 시작")
        global collection
        collection = chroma_client.create_collection(name="construction_manuals")
        logger.info("새 컬렉션 생성 완료")
        
        # FAQ 데이터를 벡터 데이터베이스에 추가
        added_count = 0
        error_count = 0
        
        logger.info("FAQ 데이터 처리 시작")
        for idx, row in faq_data.iterrows():
            try:
                logger.info(f"FAQ 항목 {idx + 1} 처리 중...")
                
                # JSON 필드 파싱
                try:
                    variations = json.loads(row['question_variations'])
                    structured_answer = json.loads(row['structured_answer'])
                    keywords = json.loads(row['keywords'])
                except json.JSONDecodeError as je:
                    logger.error(f"JSON 파싱 오류 (행 {idx}): {je}")
                    error_count += 1
                    continue
                
                # 각 질문 변형에 대해 임베딩 생성 및 저장
                for q in variations:
                    try:
                        embedding = get_embeddings(q)
                        collection.add(
                            embeddings=[embedding],
                            documents=[q],
                            metadatas=[{
                                "original_question": q,
                                "structured_answer": json.dumps(structured_answer, ensure_ascii=False),
                                "keywords": json.dumps(keywords, ensure_ascii=False)
                            }],
                            ids=[f"qa_{idx}_{variations.index(q)}"]
                        )
                        added_count += 1
                        logger.info(f"질문 변형 '{q}' 추가됨")
                    except Exception as e:
                        logger.error(f"임베딩 생성/저장 오류 (행 {idx}, 질문: {q}): {e}")
                        error_count += 1
            except Exception as e:
                error_count += 1
                logger.error(f"행 {idx} 처리 중 오류 발생: {e}")
                continue
        
        logger.info(f"벡터 데이터베이스 초기화 완료: {added_count}개 추가됨, {error_count}개 실패")
        
        # 벡터 DB 검증
        if added_count == 0:
            logger.error("벡터 데이터베이스에 데이터가 추가되지 않았습니다.")
            return False
            
        # 테스트 쿼리 실행
        test_result = collection.query(
            query_embeddings=[get_embeddings("테스트 쿼리")],
            n_results=1
        )
        
        if not test_result["ids"] or len(test_result["ids"][0]) == 0:
            logger.error("벡터 데이터베이스 검증 실패: 테스트 쿼리 결과 없음")
            return False
            
        logger.info("벡터 데이터베이스 검증 완료")
        return True
        
    except Exception as e:
        logger.error(f"벡터 데이터베이스 초기화 중 오류 발생: {e}")
        return False

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

# Excel 파일 처리 엔드포인트 추가
@app.post("/process-excel")
async def process_excel():
    """업로드된 Excel 파일을 처리하여 FAQ 데이터로 변환합니다."""
    try:
        source_path = DOCS_DIR / "qa_pairs.xlsx"
        if not source_path.exists():
            return {"status": "error", "detail": "처리할 Excel 파일이 없습니다. 먼저 파일을 업로드해 주세요."}
        
        # 외부 프로세스로 qa_allinone.py 실행
        logger.info("qa_allinone.py를 사용하여 FAQ 데이터 구조화 시작")
        try:
            # 환경 변수 설정 (Gemini API 키가 있으면 사용)
            my_env = os.environ.copy()
            
            # 만약 현재 서버에 OPENROUTER_API_KEY가 설정되어 있다면, 이를 GOOGLE_API_KEY로도 사용
            # (임시 방편, 실제로는 별도의 GOOGLE_API_KEY를 사용하는 것이 좋음)
            if OPENROUTER_API_KEY:
                my_env["GOOGLE_API_KEY"] = OPENROUTER_API_KEY
            
            # 비동기 방식으로 파이썬 스크립트 실행
            process = await asyncio.create_subprocess_exec(
                sys.executable,  # 현재 파이썬 인터프리터
                str(ROOT_DIR / "qa_allinone.py"),  # qa_allinone.py 경로
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=my_env
            )
            stdout, stderr = await process.communicate()
            
            # 결과 로깅
            if stdout:
                logger.info(f"qa_allinone.py 출력: {stdout.decode('utf-8')}")
            if stderr:
                logger.error(f"qa_allinone.py 오류: {stderr.decode('utf-8')}")
                
            if process.returncode != 0:
                logger.error(f"qa_allinone.py 실행 실패: 종료 코드 {process.returncode}")
                return {"status": "error", "detail": f"FAQ 구조화 처리 중 오류가 발생했습니다: 종료 코드 {process.returncode}"}
                
            logger.info("qa_allinone.py 실행 완료")
        except Exception as e:
            logger.error(f"qa_allinone.py 실행 중 예외 발생: {e}")
            return {"status": "error", "detail": f"FAQ 구조화 처리 중 오류가 발생했습니다: {str(e)}"}
        
        # 구조화된 FAQ 파일 확인
        if not ENHANCED_FAQ_PATH.exists():
            logger.error(f"구조화된 FAQ 파일을 찾을 수 없습니다: {ENHANCED_FAQ_PATH}")
            return {"status": "error", "detail": "FAQ 구조화 처리는 완료되었으나 결과 파일을 찾을 수 없습니다."}
        
        # 구조화된 FAQ 데이터 확인
        try:
            enhanced_df = pd.read_excel(ENHANCED_FAQ_PATH)
            enhanced_count = len(enhanced_df)
            logger.info(f"구조화된 FAQ 데이터 확인: {enhanced_count}개 항목")
        except Exception as e:
            logger.error(f"구조화된 FAQ 데이터 읽기 실패: {e}")
            enhanced_count = "알 수 없음"
        
        # FAQ 데이터 다시 로드
        load_enhanced_faq()
        
        # 벡터 DB 초기화 - 반환 전 자동으로 실행
        db_init_result = await initialize_vector_db()
        if not db_init_result:
            logger.warning("벡터 DB 초기화 중 문제가 발생했습니다. FAQ 데이터는 정상적으로 처리되었습니다.")
        
        # 원본 엑셀 파일 행 수 확인
        try:
            original_df = pd.read_excel(source_path)
            original_count = len(original_df)
        except:
            original_count = "알 수 없음"
        
        return {
            "status": "success", 
            "message": "Excel 파일이 성공적으로 처리되었습니다.",
            "details": f"원본 {original_count}개 항목 중 {enhanced_count}개가 AI를 통해 구조화되었습니다. 벡터 DB도 초기화되었습니다."
        }
    except Exception as e:
        logger.error(f"Excel 처리 중 오류: {e}")
        return {"status": "error", "detail": str(e)}

# 정적 파일 서빙 설정은 모든 API 엔드포인트 정의 후 맨 마지막에 위치
app.mount("/", StaticFiles(directory=str(FRONTEND_BUILD_DIR), html=True), name="frontend")