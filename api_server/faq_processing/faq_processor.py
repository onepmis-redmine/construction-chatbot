import json
import pandas as pd
from pathlib import Path
import re
import torch
import sys
import os
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
import vector_db

logger = get_logger(__name__)

# 환경 변수 로드
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    sys.exit(1)

try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    logger.error(f"Google API 설정 중 오류 발생: {str(e)}")
    sys.exit(1)

class FAQProcessor:
    def __init__(self, docs_dir: Path, vector_db: vector_db.VectorDB):
        self.docs_dir = docs_dir
        self.vector_db = vector_db
        self.enhanced_faq_path = docs_dir / "enhanced_qa_pairs.xlsx"
        self.faq_data = None
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
            logger.info("FAQProcessor initialized successfully")
        except Exception as e:
            logger.error(f"FAQProcessor 초기화 중 오류 발생: {str(e)}")
            sys.exit(1)

    def enhance_qa_pair(self, question: str, answer: str) -> dict:
        """Gemini를 사용하여 질문-답변 쌍을 구조화합니다."""
        prompt = f"""
다음 질문-답변 쌍을 구조화된 형식으로 변환해주세요.
결과는 반드시 올바른 JSON 형식이어야 합니다.
모든 답변은 반드시 정중하고 공손한 존댓말로 작성해주세요.
질문 변형도 가능한 존댓말로 작성하되, 다양한 표현 방식을 포함해주세요.

원본 질문: {question}
원본 답변: {answer}

다음 구조로 JSON을 생성해주세요:
{{
    "question_variations": [
        "변형된 질문 1 (존댓말로 작성)",
        "변형된 질문 2 (존댓말로 작성)",
        "변형된 질문 3 (존댓말로 작성)"
    ],
    "structured_answer": {{
        "basic_rules": [
            "기본 규칙/조건 1 (존댓말로 작성)",
            "기본 규칙/조건 2 (존댓말로 작성)"
        ],
        "examples": [
            {{
                "scenario": "예시 상황 설명 (존댓말로 작성)",
                "result_a": "A의 결과 (존댓말로 작성)",
                "result_b": "B의 결과 (존댓말로 작성)"
            }},
            {{
                "scenario": "다른 예시 상황 (존댓말로 작성)",
                "result_c": "C의 결과 (존댓말로 작성)"
            }}
        ],
        "cautions": [
            "주의사항 1 (존댓말로 작성)",
            "주의사항 2 (존댓말로 작성)"
        ]
    }},
    "keywords": [
        "키워드1",
        "키워드2",
        "키워드3",
        "키워드4",
        "키워드5",
        "키워드6"
    ]
}}

특별 지시사항:
1. keywords 항목에는 최소 6개 이상의 키워드를 포함해주세요.
2. 한글에 포함된 한자어와 그 순우리말 동의어를 모두 키워드에 포함해주세요. (예: '근로자'와 '일꾼', '휴일'과 '쉬는 날')
3. 띄어쓰기 유무에 따른 변형도 별도 키워드로 포함해주세요. (예: '주휴수당'과 '주휴 수당')
4. 유사 개념이지만 다른 의미의 키워드도 포함해주세요. (예: '주휴수당'과 '휴일수당'은 구분)
5. 실무에서 흔히 쓰이는 줄임말이나 약어도 포함해주세요. (예: '퇴직금'과 '퇴직금여')
6. 질문에 포함된 핵심 단어는 반드시 모두 키워드에 포함해주세요.

응답은 반드시 위와 같은 형식의 유효한 JSON이어야 합니다.
모든 설명과 답변은 반드시 정중하고 공손한 존댓말로 작성해주세요.
특히 주휴수당 관련 예시의 경우, A와 B는 한 예시에서 함께 설명하고, C는 별도의 예시로 구분해주세요.
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
            logger.error(f"Error enhancing Q&A pair: {str(e)}")
            return None

    async def process_faq(self) -> bool:
        """업로드된 FAQ 파일을 처리하고 구조화합니다."""
        try:
            # 업로드된 파일 찾기
            upload_dir = self.docs_dir.parent / "uploads"
            excel_files = list(upload_dir.glob("*.xlsx"))
            
            if not excel_files:
                logger.error("처리할 Excel 파일을 찾을 수 없습니다.")
                return False
            
            # 가장 최근 파일 선택
            latest_file = max(excel_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"처리할 파일: {latest_file}")
            
            # Excel 파일 읽기
            df = pd.read_excel(latest_file)
            
            # 필수 컬럼 확인
            required_columns = ['질문', '답변']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"필수 컬럼이 없습니다. 필요한 컬럼: {required_columns}")
                return False
            
            # 구조화된 FAQ 데이터 생성
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
                return False

            # 결과를 새로운 엑셀 파일로 저장
            self.faq_data = pd.DataFrame(enhanced_data)
            self.faq_data.to_excel(self.enhanced_faq_path, index=False)
            logger.info(f"Successfully saved enhanced FAQ to: {self.enhanced_faq_path}")
            
            # 벡터 DB 초기화 (force_rebuild=True로 설정하여 기존 컬렉션 재구축)
            success = await self.vector_db.initialize_vector_db(self.faq_data, force_rebuild=True)
            if not success:
                logger.error("벡터 DB 초기화 실패")
                return False
            
            logger.info("FAQ 처리 완료")
            return True
        except Exception as e:
            logger.error(f"FAQ 처리 중 오류 발생: {e}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
            return False

    def load_enhanced_faq(self):
        """구조화된 FAQ 데이터를 로드합니다."""
        try:
            if not self.enhanced_faq_path.exists():
                logger.error(f"Enhanced FAQ file not found at: {self.enhanced_faq_path}")
                return False
                
            self.faq_data = pd.read_excel(self.enhanced_faq_path)
            
            if self.faq_data.empty:
                logger.error("Enhanced FAQ file is empty")
                return False
            
            # 데이터 컬럼 확인 로깅
            logger.info(f"FAQ 데이터 컬럼: {list(self.faq_data.columns)}")
            logger.info(f"FAQ 데이터 첫 행: \n{self.faq_data.iloc[0]}")
            
            # 데이터 형식 검증
            for col in ['question_variations', 'structured_answer', 'keywords']:
                if col not in self.faq_data.columns:
                    logger.error(f"Required column '{col}' not found in FAQ data")
                    return False
            
            # JSON 필드 파싱 검증
            for idx, row in self.faq_data.iterrows():
                try:
                    for field in ['question_variations', 'structured_answer', 'keywords']:
                        if isinstance(row[field], str):
                            json.loads(row[field])
                        else:
                            logger.warning(f"Row {idx}: {field} is not a string, converting to string")
                            self.faq_data.at[idx, field] = json.dumps(row[field], ensure_ascii=False)
                except Exception as e:
                    logger.error(f"Error parsing JSON in row {idx}: {e}")
                    return False
                
            logger.info(f"Enhanced FAQ data loaded successfully: {len(self.faq_data)} entries found")
            return True
        except Exception as e:
            logger.error(f"Error loading enhanced FAQ: {e}")
            self.faq_data = None
            return False

    def find_faq_match(self, query: str, threshold: float = 0.75):
        """구조화된 FAQ에서 가장 적절한 답변을 찾습니다."""
        
        if self.faq_data is None or self.faq_data.empty:
            logger.error("FAQ data is not loaded")
            return None, 0.0
        
        try:
            # 쿼리 전처리
            query = query.lower().strip()
            if query.endswith('?') or query.endswith('.'): 
                query = query[:-1]
            
            # 쿼리 임베딩 생성
            query_embedding = self.vector_db.get_embeddings(query)
            
            # 단일 임베딩인 경우 리스트로 변환
            if isinstance(query_embedding, list) and not isinstance(query_embedding[0], list):
                query_embedding = [query_embedding]
                
            # 만약 임베딩 결과가 빈 리스트이거나 None이면 기본값 사용
            if not query_embedding or len(query_embedding) == 0:
                logger.error("임베딩 생성 실패, 기본값 사용")
                # 384차원 0 벡터 사용
                query_embedding = [[0.0] * 384]
            
            # 컬렉션 가져오기
            current_collection = self.vector_db.get_collection()
            
            logger.info(f"벡터 검색 시작: {query}")
            
            # 기존 방식으로 돌아가기: 모든 FAQ 항목을 직접 비교
            best_match = None
            highest_similarity = -1
            
            # 쿼리에 포함된 단어 목록
            query_words = set(query.split())
            
            # 각 FAQ 항목과 비교
            for idx, row in self.faq_data.iterrows():
                try:
                    variations = json.loads(row['question_variations']) if isinstance(row['question_variations'], str) else row['question_variations']
                    original_question = row.get('original_question', '')
                    
                    # keywords 필드 활용
                    keywords = []
                    if 'keywords' in row:
                        try:
                            keywords_data = row['keywords']
                            if isinstance(keywords_data, str):
                                keywords = json.loads(keywords_data)
                            else:
                                keywords = keywords_data
                        except Exception as e:
                            logger.error(f"키워드 파싱 오류: {e}")
                    
                    logger.info(f"FAQ 항목 {idx + 1}/{len(self.faq_data)}: 원본 질문: '{original_question}', 키워드: {keywords}")
                    
                    # 키워드 매칭 점수 계산
                    keyword_bonus = 0.0
                    matching_keywords = []
                    
                    for keyword in keywords:
                        # 정확한 키워드 매칭 확인 (부분 매칭이 아닌 완전 일치)
                        # 1. 쿼리에 키워드가 정확히 포함되는지 확인 (단어 경계 고려)
                        exact_match = False
                        
                        # 키워드를 소문자로 변환하고 공백 처리
                        keyword_lower = keyword.lower().strip()
                        # 띄어쓰기를 제거한 버전도 생성
                        keyword_no_space = keyword_lower.replace(" ", "")
                        
                        # 쿼리에서 띄어쓰기 제거한 버전
                        query_no_space = query.lower().replace(" ", "")
                        # 쿼리 단어를 모두 붙인 버전
                        query_words_joined = "".join(query_words)
                        
                        # 방법 1: 쿼리에서 키워드를 단어 단위로 찾기
                        if keyword_lower in query_words:
                            exact_match = True
                            logger.info(f"단어 단위 매치: '{keyword}'")
                        
                        # 방법 2: 쿼리에서 키워드를 단어 경계를 고려하여 찾기
                        elif re.search(r'\b' + re.escape(keyword_lower) + r'\b', query.lower()):
                            exact_match = True
                            logger.info(f"단어 경계 매치: '{keyword}'")
                        
                        # 방법 3: 쿼리가 기본적으로 짧은 경우 정확한 키워드만 체크
                        elif len(query_words) <= 3 and query.lower() == keyword_lower:
                            exact_match = True
                            logger.info(f"전체 텍스트 매치: '{keyword}'")
                        
                        # 방법 4: 띄어쓰기를 무시하고 매칭
                        elif keyword_no_space == query_no_space:
                            exact_match = True
                            logger.info(f"띄어쓰기 무시 전체 매치: '{keyword}' vs '{query}'")
                        
                        # 방법 5: 띄어쓰기를 무시한 부분 문자열 매칭 (키워드가 꽤 길 경우에만)
                        elif len(keyword_no_space) >= 4 and keyword_no_space in query_no_space:
                            # 키워드가 충분히 길 경우에만 부분 문자열 매칭 허용
                            exact_match = True
                            logger.info(f"띄어쓰기 무시 부분 매치: '{keyword}' in '{query}'")
                        
                        if exact_match:
                            matching_keywords.append(keyword)
                            keyword_bonus += 0.3  # 키워드당 0.3점 보너스
                            logger.info(f"정확한 키워드 매치: '{keyword}'")
                    
                    if matching_keywords:
                        logger.info(f"키워드 매치 발견: {matching_keywords}, 보너스: {keyword_bonus}")
                    
                    # 현재 질문과의 유사도 계산
                    current_question_similarity = 0.0
                    try:
                        current_embedding = self.vector_db.get_embeddings(original_question)
                        if isinstance(current_embedding, list) and not isinstance(current_embedding[0], list):
                            current_embedding = [current_embedding]
                        current_question_similarity = torch.cosine_similarity(
                            torch.tensor(query_embedding[0]),
                            torch.tensor(current_embedding[0]),
                            dim=0
                        ).item()
                        # 현재 질문에 1.5배 가중치 적용
                        current_question_similarity *= 1.5
                        logger.info(f"현재 질문 유사도 (가중치 적용): {current_question_similarity:.4f}")
                    except Exception as e:
                        logger.error(f"현재 질문 유사도 계산 중 오류: {e}")
                    
                    for q in variations:
                        # 질문 전처리
                        q = q.lower().strip()
                        if q.endswith('?') or q.endswith('.'): 
                            q = q[:-1]
                            
                        # 직접 코사인 유사도 계산
                        try:
                            # q에 대한 임베딩 조회
                            q_embedding = self.vector_db.get_embeddings(q)
                            
                            # 단일 임베딩인 경우 리스트로 변환
                            if isinstance(q_embedding, list) and not isinstance(q_embedding[0], list):
                                q_embedding = [q_embedding]
                            
                            # 코사인 유사도 계산
                            similarity = torch.cosine_similarity(
                                torch.tensor(query_embedding[0]),
                                torch.tensor(q_embedding[0]),
                                dim=0
                            ).item()
                            
                            # 현재 질문 유사도와 변형 질문 유사도 중 높은 값 선택
                            max_similarity = max(similarity, current_question_similarity)
                            
                            # 키워드 보너스 적용
                            adjusted_similarity = max_similarity + keyword_bonus
                            
                            logger.info(f"질문: '{q}', 기본 유사도: {similarity:.4f}, 현재 질문 유사도: {current_question_similarity:.4f}, 키워드 보너스: {keyword_bonus:.2f}, 최종: {adjusted_similarity:.4f}")
                            
                            if adjusted_similarity > highest_similarity:
                                highest_similarity = adjusted_similarity
                                best_match = row
                                logger.info(f"새 최고 매치: '{original_question}', 유사도: {adjusted_similarity:.4f}, 매칭 키워드: {matching_keywords}")
                        except Exception as e:
                            logger.error(f"유사도 계산 중 오류: {e}")
                            continue
                except Exception as e:
                    logger.error(f"변형 질문 처리 중 오류: {e}")
                    continue
            
            # 최종 유사도가 키워드 보너스 때문에 임계값을 넘었을 수 있으므로, 
            # 원래 임계값보다 크거나 같은지 확인
            if highest_similarity >= threshold and best_match is not None:
                return best_match, highest_similarity
            return None, highest_similarity if highest_similarity > -1 else 0.0
        except Exception as e:
            logger.error(f"Error in find_faq_match: {e}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
            return None, 0.0 