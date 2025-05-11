from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer
import httpx
import json
from fastapi.middleware.cors import CORSMiddleware #frontend React 프로젝트 추가
import logging

#그록3 로깅 설정
logging.basicConfig(
    filename="query_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# OpenRouter API Key
OPENROUTER_API_KEY = "Bearer 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"

# FastAPI 앱 생성
app = FastAPI()

#frontend React 프로젝트 추가 시작
# CORS 설정 (React 개발 서버에서 오는 요청 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#frontend React 프로젝트 추가 끝



# ChromaDB 클라이언트 설정
chroma_client = chromadb.PersistentClient(path="../vector_db")
collection = chroma_client.get_or_create_collection(name="construction_manuals")

# 임베딩 모델 로드
embedder = SentenceTransformer("all-MiniLM-L6-v2")

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
        # ,
        # "temperature": 0.2, # 답변이 더 보수적이고 프롬프트에 충실
        # "max_tokens": 200 # 답변이 너무 길어지면 문맥을 벗어날 가능성이 있

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

# API 엔드포인트
@app.post("/ask")
def ask_question(q: Question):
    query = q.query
    query_embedding = embedder.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    # 출처 포함된 텍스트 구성
    retrieved_info = ""
    sources = []  # sources 리스트 초기화
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        source_info = meta.get("source", "알 수 없는 출처") if meta else "알 수 없는 출처"
        retrieved_info += f"{doc}\n[출처: {source_info}]\n\n"
        sources.append(source_info)  # 출처 리스트에 추가
    

    # GPT 프롬프트 구성
    prompt = f"""다음은 시스템 매뉴얼의 일부입니다. 이 내용을 참고하여 사용자의 질문에 정확히 답변해 주세요. 매뉴얼에서 답을 찾았다면 바로 보여주세요.


[참고 문서]
{retrieved_info}

[사용자 질문]
{query}

[답변]
"""

    answer = call_openrouter(prompt)
    
    # 로깅 추가: 질문, 답변, 출처 기록
    logging.info(f"Question: {query} | Answer: {answer} | Sources: {sources}")

    # ✅ 콘솔에 출력
    print("=== 사용자 질문 ===")
    print(query)
    print("=== GPT 응답 ===")
    print(answer)
    print("=== 출처 ===")
    print(sources)

    return {
        "answer": answer,
        "sources": sources  # 리스트로 반환
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}