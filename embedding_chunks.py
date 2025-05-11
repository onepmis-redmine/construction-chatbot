#코파일럿이 추천해준 수정된 코드
import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
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

# 임베딩 모델 로드
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 문단 읽고 임베딩해서 저장
batch_size = 32  # ✅ 배치 처리로 메모리 최적화
total_added = 0

for filename in tqdm(os.listdir(CHUNKS_DIR)):
    if not filename.endswith(".json"):
        continue

    filepath = os.path.join(CHUNKS_DIR, filename)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except json.JSONDecodeError:
        print(f"❌ {filename} 파일이 올바른 JSON 형식이 아닙니다.")
        continue

    texts = [chunk.get("text", "").strip() for chunk in chunks if chunk.get("text")]
    ids = [f"{filename}_{i}" for i in range(len(texts))]
    metadatas = [{"source": filename}] * len(texts)


    # ✅ 빈 리스트 예외 처리
    if not texts:
        print(f"⚠️ {filename}에서 저장할 데이터가 없습니다.")
        continue

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        embeddings.extend(embedder.encode(batch_texts).tolist())

    if not (len(texts) == len(embeddings) == len(ids) == len(metadatas)):
        print(f"❌ 데이터 길이 불일치: texts={len(texts)}, embeddings={len(embeddings)}, ids={len(ids)}, metadatas={len(metadatas)}")
        continue

    try:
        collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        total_added += len(texts)        
    except Exception as e:
        print(f"❌ 벡터DB 저장 중 오류 발생: {str(e)}")

print(f"✅ 모든 임베딩 저장 완료! 총 추가된 문서 수: {total_added}")