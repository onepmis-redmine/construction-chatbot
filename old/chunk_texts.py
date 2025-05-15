import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

# 폴더 경로
RAW_TEXTS_DIR = "raw_texts/"
CHUNKS_DIR = "chunks/"

# 저장 폴더 없으면 만들기
os.makedirs(CHUNKS_DIR, exist_ok=True)

# 텍스트 쪼개기 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # 한 덩어리 크기
    chunk_overlap=50,     # 겹치는 문자 수 (문맥 유지)
    length_function=len
)

# 파일별로 쪼개기
for filename in tqdm(os.listdir(RAW_TEXTS_DIR)):
    if not filename.endswith(".txt"):
        continue

    filepath = os.path.join(RAW_TEXTS_DIR, filename)

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # 쪼개기
    chunks = text_splitter.split_text(text)

    # 저장
    output = [{"text": chunk} for chunk in chunks]
    output_path = os.path.join(CHUNKS_DIR, filename.replace(".txt", ".json"))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
