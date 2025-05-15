#코파일럿이 추천해준 수정된 코드
import os
import json
import chromadb
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from pathlib import Path


# 폴더 경로
CHUNKS_DIR = "chunks/"
DB_DIR = "vector_db/"


# 이 부분 추가!
chunk_dir = Path("chunks")
chunk_paths = sorted(chunk_dir.glob("*.txt"))


# 🛠 폴더 존재 여부 확인
if not os.path.exists(CHUNKS_DIR):
    print(f"⚠️ 폴더 '{CHUNKS_DIR}'가 존재하지 않습니다.")
    exit(1)

os.makedirs(DB_DIR, exist_ok=True)  # 벡터DB 저장할 폴더 생성

# ChromaDB 클라이언트 생성
chroma_client = chromadb.PersistentClient(path=DB_DIR)

# 벡터DB Collection 생성
collection = chroma_client.get_or_create_collection(name="construction_manuals")

# 임베딩 모델과 토크나이저 로드
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Mean Pooling 함수
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# 텍스트를 임베딩으로 변환하는 함수
def get_embeddings(texts):
    # 입력이 문자열인 경우 리스트로 변환
    if isinstance(texts, str):
        texts = [texts]
    
    # 토크나이즈 및 모델 입력 준비
    encoded_input = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    
    # 모델 추론
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Mean Pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    
    # 정규화
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    
    return sentence_embeddings[0].tolist() if len(texts) == 1 else sentence_embeddings.tolist()

# 문단 읽고 임베딩해서 저장
batch_size = 32  # ✅ 배치 처리로 메모리 최적화
total_added = 0

for filename in tqdm(os.listdir(CHUNKS_DIR)):
    if not filename.endswith(".json"):
        continue

    filepath = os.path.join(CHUNKS_DIR, filename)

    with open(filepath, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [chunk["text"] for chunk in chunks]
    ids = [f"{filename}_{i}" for i in range(len(texts))]

    # 배치 처리
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        
        embeddings = get_embeddings(batch_texts)
        
        collection.add(
            documents=batch_texts,
            embeddings=embeddings,
            ids=batch_ids,
            metadatas=[{"source": filename}] * len(batch_texts)
        )
        
        total_added += len(batch_texts)

print(f"✅ 임베딩 저장 완료! 총 {total_added}개의 문단이 처리되었습니다.")