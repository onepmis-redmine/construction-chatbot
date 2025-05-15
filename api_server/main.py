from fastapi import FastAPI, Request
from pydantic import BaseModel
import chromadb
import torch
from transformers import AutoTokenizer, AutoModel
import httpx
import json
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from dotenv import load_dotenv
import gc
from pathlib import Path
import pandas as pd
from fastapi.staticfiles import StaticFiles

# 프로젝트 루트 디렉토리 설정
ROOT_DIR = Path(__file__).parent.parent
VECTOR_DB_PATH = ROOT_DIR / "vector_db"
DOCS_DIR = ROOT_DIR / "docs"
ENHANCED_FAQ_PATH = DOCS_DIR / "enhanced_qa_pairs.xlsx"
FRONTEND_BUILD_DIR = ROOT_DIR / "frontend" / "build"

# 디렉토리 존재 확인 및 생성
VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()  # .env 파일 읽기

# 환경 변수에서 API 키 불러오기
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

#그록3 로깅 설정
logging.basicConfig(
    filename=ROOT_DIR / "query_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 콘솔 출력을 위한 핸들러 추가
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

# 모델 캐시 디렉토리 설정
cache_dir = ROOT_DIR / "model_cache"
cache_dir.mkdir(parents=True, exist_ok=True)

# FastAPI 앱 생성
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://construction-chatbot-api.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            logging.error(f"Enhanced FAQ file not found at: {ENHANCED_FAQ_PATH}")
            return False
            
        faq_data = pd.read_excel(ENHANCED_FAQ_PATH)
        
        if faq_data.empty:
            logging.error("Enhanced FAQ file is empty")
            return False
        
        # 데이터 형식 검증
        for col in ['question_variations', 'structured_answer', 'keywords']:
            if col not in faq_data.columns:
                logging.error(f"Required column '{col}' not found in FAQ data")
                return False
        
        # JSON 필드 파싱 검증
        for idx, row in faq_data.iterrows():
            try:
                for field in ['question_variations', 'structured_answer', 'keywords']:
                    if isinstance(row[field], str):
                        json.loads(row[field])
                    else:
                        logging.warning(f"Row {idx}: {field} is not a string, converting to string")
                        faq_data.at[idx, field] = json.dumps(row[field], ensure_ascii=False)
            except Exception as e:
                logging.error(f"Error parsing JSON in row {idx}: {e}")
                return False
            
        logging.info(f"Enhanced FAQ data loaded successfully: {len(faq_data)} entries found")
        return True
    except Exception as e:
        logging.error(f"Error loading enhanced FAQ: {e}")
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
        logging.error("FAQ data is not loaded")
        return None
    
    try:
        logging.info(f"Searching for match to query: {query}")
        logging.info(f"Available FAQ entries: {len(faq_data)}")
        
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
                logging.info(f"Row {idx} variations: {variations}")
                
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
                    
                    logging.info(f"Comparing '{query}' with '{q}': similarity = {similarity}")
                    
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        best_match = row
                        logging.info(f"New best match found: similarity = {similarity}")
            except Exception as e:
                logging.error(f"Error processing row {idx}: {e}")
                continue
        
        logging.info(f"Best match similarity: {highest_similarity} (threshold: {threshold})")
        if highest_similarity >= threshold:
            return best_match
        return None
    except Exception as e:
        logging.error(f"Error in find_faq_match: {e}")
        return None

@app.post("/ask")
async def ask_question(q: Question):
    query = q.query.strip()
    logging.info(f"Received question: {query}")
    
    try:
        # 벡터 검색 수행
        query_embedding = get_embeddings(query)
        logging.info("Generated query embedding")
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )
        
        logging.info(f"Vector search results: {results}")
        
        # 결과가 비어있는 경우 처리
        if not results["ids"] or len(results["ids"][0]) == 0:
            logging.warning("No results found in vector database")
            return {
                "answer": "죄송합니다. 해당 질문에 대한 답변을 찾을 수 없습니다. 데이터베이스에 데이터가 아직 로드되지 않았을 수 있습니다.",
                "sources": [],
                "is_faq": False
            }
            
        # 가장 유사한 문서 찾기
        best_match_idx = 0
        best_match_distance = results["distances"][0][best_match_idx]
        
        logging.info(f"Best match distance: {best_match_distance}")
        
        if best_match_distance > 1.5:  # 거리가 너무 멀면 관련이 없는 것으로 판단
            logging.warning(f"Best match distance ({best_match_distance}) too high")
            return {
                "answer": "죄송합니다. 해당 질문과 충분히 관련된 답변을 찾지 못했습니다.",
                "sources": [],
                "is_faq": False
            }
            
        # 메타데이터에서 구조화된 답변 추출
        metadata = results["metadatas"][0][best_match_idx]
        structured_answer = json.loads(metadata["structured_answer"])
        
        # 텍스트 기반 응답 생성
        answer_parts = []
        
        # 기본 규칙 섹션
        if structured_answer['basic_rules']:
            answer_parts.append("[기본 규칙]\n")
            for rule in structured_answer['basic_rules']:
                answer_parts.append(f"• {rule}\n")
        
        # 예시 섹션
        if structured_answer['examples']:
            answer_parts.append("\n[예시]\n")
            for example in structured_answer['examples']:
                if isinstance(example, dict):
                    # 딕셔너리 형태의 예시를 읽기 쉽게 포맷팅
                    scenario = example.get('scenario', '')
                    result = example.get('result', '')
                    explanation = example.get('explanation', '')
                    formatted_example = f"• {scenario}\n  → {result}\n  ☞ {explanation}"
                    answer_parts.append(f"{formatted_example}\n")
                else:
                    # 일반 문자열 형태의 예시
                    answer_parts.append(f"• {example}\n")
        
        # 주의사항 섹션
        if structured_answer['cautions']:
            answer_parts.append("\n[주의사항]\n")
            for caution in structured_answer['cautions']:
                answer_parts.append(f"⚠️ {caution}\n")
        
        # 모든 섹션을 결합하고 불필요한 공백 제거
        answer = "".join(answer_parts).strip()
        
        return {
            "answer": answer,
            "sources": ["FAQ 데이터베이스"],
            "is_faq": True,
            "original_question": metadata.get("original_question", "Unknown"),
            "match_distance": best_match_distance
        }
        
    except Exception as e:
        logging.error(f"Error processing question: {e}")
        return {
            "answer": "죄송합니다. 질문 처리 중 오류가 발생했습니다.",
            "sources": [],
            "is_faq": False,
            "error": str(e)
        }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/reload-faq")
async def reload_faq():
    """FAQ 데이터를 재로드합니다."""
    load_enhanced_faq()
    return {"status": "FAQ reloaded"}

@app.get("/init-db")
async def initialize_database():
    """벡터 데이터베이스를 초기화합니다."""
    try:
        # FAQ 데이터 로드
        if not load_enhanced_faq():
            return {"status": "error", "message": "FAQ 데이터 로드 실패"}
        
        # 기존 컬렉션 삭제 후 재생성
        try:
            chroma_client.delete_collection(name="construction_manuals")
        except:
            pass
        
        collection = chroma_client.create_collection(name="construction_manuals")
        
        # FAQ 데이터를 벡터 데이터베이스에 추가
        for idx, row in faq_data.iterrows():
            try:
                variations = json.loads(row['question_variations'])
                structured_answer = json.loads(row['structured_answer'])
                keywords = json.loads(row['keywords'])
                
                # 각 질문 변형에 대해 임베딩 생성 및 저장
                for q in variations:
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
            except Exception as e:
                logging.error(f"Error processing row {idx}: {e}")
                continue
        
        return {"status": "success", "message": "데이터베이스가 성공적으로 초기화되었습니다."}
    except Exception as e:
        logging.error(f"Error initializing database: {e}")
        return {"status": "error", "message": str(e)}

# 마지막으로 정적 파일 서빙 설정
app.mount("/", StaticFiles(directory=str(FRONTEND_BUILD_DIR), html=True), name="frontend")