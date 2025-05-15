import os
import json
import pandas as pd
from pathlib import Path
import logging
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm
import sys
import codecs

# Windows 콘솔 출력 인코딩 설정
if sys.platform.startswith('win'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 프로젝트 루트 디렉토리 설정
ROOT_DIR = Path(__file__).parent
DOCS_DIR = ROOT_DIR / "docs"
DB_DIR = ROOT_DIR / "vector_db"
EXCEL_PATH = DOCS_DIR / "qa_pairs.xlsx"
ENHANCED_FAQ_PATH = DOCS_DIR / "enhanced_qa_pairs.xlsx"
LOG_DIR = ROOT_DIR / "logs"

# 필요한 디렉토리 생성
DOCS_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 환경 변수 로드
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# 로깅 설정
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_file_handler = logging.FileHandler(LOG_DIR / 'qa_enhancement.log', encoding='utf-8')
log_file_handler.setFormatter(log_formatter)
log_stream_handler = logging.StreamHandler(sys.stdout)
log_stream_handler.setFormatter(log_formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(log_file_handler)
logger.addHandler(log_stream_handler)

# ChromaDB 및 임베딩 모델 초기화
chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
collection = chroma_client.get_or_create_collection(name="construction_manuals")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

class QAProcessor:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
        logger.info("QAProcessor initialized successfully")

    def read_qa_excel(self) -> pd.DataFrame:
        """FAQ 엑셀 파일을 읽어옵니다."""
        try:
            df = pd.read_excel(EXCEL_PATH)
            logger.info(f"Successfully read FAQ file: {EXCEL_PATH}")
            return df
        except Exception as e:
            logger.error(f"Error reading FAQ file: {e}")
            raise

    def enhance_qa_pair(self, question: str, answer: str) -> dict:
        """Gemini를 사용하여 질문-답변 쌍을 구조화합니다."""
        prompt = f"""
다음 질문-답변 쌍을 구조화된 형식으로 변환해주세요.
결과는 반드시 올바른 JSON 형식이어야 합니다.

원본 질문: {question}
원본 답변: {answer}

주어진 답변에서 다음 내용을 추출하여 JSON으로 구조화해주세요:
1. 기본 규칙/조건은 답변의 주요 내용에서 추출
2. 예시는 답변 내용 중 "예시", "예를 들어", "예:" 등으로 시작하는 부분에서 추출
   - 만약 근무시간이나 수당 계산과 관련된 내용이라면, 구체적인 시간과 계산 예시를 포함해주세요
   - 예: "주 49시간 근무 시 → 기본 40시간 + 연장 9시간"
   - 예: "토요일 8시간 근무 시 → 주간 연장수당 8시간"
3. 주의사항은 답변 내용 중 "주의", "유의", "참고" 등으로 시작하는 부분에서 추출
4. 만약 답변에서 예시나 주의사항이 명시적으로 언급되지 않은 경우, 적절한 계산 예시나 주의사항을 생성해주세요

다음 구조로 JSON을 생성해주세요:
{{
    "question_variations": [
        "원본 질문과 동일한 의미의 다양한 표현 1 (구어체로)",
        "원본 질문과 동일한 의미의 다양한 표현 2 (문어체로)",
        "원본 질문과 동일한 의미의 다양한 표현 3 (간단한 표현)",
        "원본 질문과 동일한 의미의 다양한 표현 4 (자세한 표현)",
        "원본 질문과 동일한 의미의 다양한 표현 5 (다른 관점)"
    ],
    "structured_answer": {{
        "basic_rules": [
            "답변의 주요 내용에서 추출한 기본 규칙/조건"
        ],
        "examples": [
            "구체적인 수치를 포함한 예시 (근무시간, 계산방법 등)",
            "실제 상황에 기반한 구체적인 사례"
        ],
        "cautions": [
            "답변 내용에서 추출한 주의사항 (있는 경우에만)",
            "규정이나 계산 시 주의할 점"
        ]
    }},
    "keywords": [
        "답변 내용에서 추출한 주요 키워드 1",
        "답변 내용에서 추출한 주요 키워드 2",
        "답변 내용에서 추출한 주요 키워드 3",
        "답변 내용에서 추출한 주요 키워드 4",
        "답변 내용에서 추출한 주요 키워드 5"
    ]
}}

응답은 반드시 위와 같은 형식의 유효한 JSON이어야 하며, question_variations에는 다양한 표현 방식의 질문이 포함되어야 합니다.
각 변형된 질문은 원본 질문의 의미는 유지하되, 다른 방식으로 표현해야 합니다.
특히 근무시간과 수당 계산이 포함된 경우, examples에 구체적인 시간과 계산 방법을 포함해주세요.
"""
        try:
            response = self.model.generate_content(prompt)
            json_str = response.text
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].strip()
            
            enhanced_content = json.loads(json_str)
            logger.info(f"Successfully enhanced Q&A pair: {question}")
            return enhanced_content
        except Exception as e:
            logger.error(f"Error enhancing Q&A pair: {e}")
            return None

    def process_and_enhance_qa(self):
        """FAQ를 읽고 구조화합니다."""
        df = self.read_qa_excel()
        enhanced_data = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="FAQ 구조화 중"):
            question = str(row['질문']).strip()
            answer = str(row['답변']).strip()
            
            enhanced = self.enhance_qa_pair(question, answer)
            if enhanced:
                # 데이터 검증
                required_fields = ['question_variations', 'structured_answer', 'keywords']
                if all(field in enhanced for field in required_fields):
                    # JSON 필드를 문자열로 변환
                    enhanced = {
                        'original_question': question,
                        'question_variations': json.dumps(enhanced['question_variations'], ensure_ascii=False),
                        'structured_answer': json.dumps(enhanced['structured_answer'], ensure_ascii=False),
                        'keywords': json.dumps(enhanced['keywords'], ensure_ascii=False)
                    }
                    enhanced_data.append(enhanced)
                    logger.info(f"Successfully processed Q&A pair {idx + 1}: {question}")
                else:
                    logger.error(f"Missing required fields in enhanced data for question: {question}")
                
            logger.info(f"Processed {idx + 1}/{len(df)} QA pairs")

        if not enhanced_data:
            logger.error("No FAQ data was successfully processed")
            return []

        # 결과를 새로운 엑셀 파일로 저장
        output_df = pd.DataFrame(enhanced_data)
        
        # 저장 전 데이터 확인
        logger.debug(f"Enhanced data before saving:\n{output_df.to_string()}")
        
        try:
            output_df.to_excel(ENHANCED_FAQ_PATH, index=False)
            logger.info(f"Successfully saved enhanced FAQ to: {ENHANCED_FAQ_PATH}")
        except Exception as e:
            logger.error(f"Error saving enhanced FAQ: {e}")
            
        return enhanced_data

    def create_embeddings(self, enhanced_data):
        """구조화된 FAQ 데이터를 벡터 데이터베이스에 임베딩합니다."""
        try:
            # 기존 컬렉션 삭제 후 재생성
            logger.info("Recreating collection...")
            try:
                chroma_client.delete_collection(name="construction_manuals")
            except:
                pass
            collection = chroma_client.create_collection(name="construction_manuals")
            
            texts = []
            metadatas = []
            ids = []

            for idx, item in enumerate(enhanced_data):
                # JSON 문자열을 파싱
                question_variations = json.loads(item['question_variations'])
                structured_answer = json.loads(item['structured_answer'])
                keywords = json.loads(item['keywords'])
                original_question = item['original_question']

                logger.info(f"Processing FAQ item {idx + 1}:")
                logger.info(f"Original question: {original_question}")
                logger.info(f"Variations: {question_variations}")

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
                answer_parts = []
                if structured_answer['basic_rules']:
                    answer_parts.extend([f"기본 규칙: {rule}" for rule in structured_answer['basic_rules']])
                if structured_answer['examples']:
                    answer_parts.extend([f"예시: {example}" for example in structured_answer['examples']])
                if structured_answer['cautions']:
                    answer_parts.extend([f"주의사항: {caution}" for caution in structured_answer['cautions']])

                for part_idx, part in enumerate(answer_parts):
                    texts.append(part)
                    metadatas.append({
                        "type": "answer_part",
                        "original_question": original_question,
                        "structured_answer": json.dumps(structured_answer, ensure_ascii=False),
                        "keywords": json.dumps(keywords, ensure_ascii=False)
                    })
                    ids.append(f"a_{idx}_{part_idx}")

            logger.info(f"Created {len(texts)} items for embedding")
            logger.debug(f"Texts to embed: {texts}")
            logger.debug(f"Metadata: {metadatas}")
            logger.debug(f"IDs: {ids}")

            # 임베딩 생성
            logger.info("Generating embeddings...")
            embeddings = embedder.encode(texts, show_progress_bar=True)
            
            # ChromaDB에 데이터 추가
            logger.info("Adding data to ChromaDB...")
            collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully embedded {len(texts)} items in the vector database")
            
            # 벡터 데이터베이스 내용 확인
            result = collection.query(
                query_texts=["테스트 쿼리"],
                n_results=1
            )
            logger.info(f"Vector DB test query result: {result}")
            
        except Exception as e:
            logger.error(f"Error in create_embeddings: {e}")
            raise

def main():
    processor = QAProcessor()
    
    # FAQ 구조화
    logger.info("Starting FAQ enhancement process...")
    enhanced_data = processor.process_and_enhance_qa()
    
    # 벡터 데이터베이스에 임베딩
    logger.info("Starting embedding process...")
    processor.create_embeddings(enhanced_data)
    
    logger.info("Process completed successfully!")

if __name__ == "__main__":
    main() 