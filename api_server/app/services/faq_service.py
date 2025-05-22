import pandas as pd
import json
from pathlib import Path
from utils.logger import get_logger
from .embedding import EmbeddingService
from .vector_db import VectorDBService

logger = get_logger(__name__)

class FAQService:
    def __init__(self, vector_db_service: VectorDBService, embedding_service: EmbeddingService, enhanced_faq_path: Path = None):
        self.vector_db_service = vector_db_service
        self.embedding_service = embedding_service
        self.enhanced_faq_path = enhanced_faq_path
        self.faq_data = None

    def load_enhanced_faq(self):
        """구조화된 FAQ 데이터를 로드합니다."""
        try:
            if not self.enhanced_faq_path or not self.enhanced_faq_path.exists():
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

    def process_faq_data(self):
        """FAQ 데이터를 처리하고 벡터 DB에 저장합니다."""
        try:
            if self.faq_data is None or self.faq_data.empty:
                logger.error("FAQ 데이터가 비어있습니다.")
                return False
            
            # 벡터 DB 초기화
            self.vector_db_service.initialize()
            
            # 각 FAQ 항목에 대해 임베딩 생성 및 저장
            for idx, row in self.faq_data.iterrows():
                try:
                    # 질문 변형들을 하나의 문자열로 결합
                    question_variations = json.loads(row['question_variations'])
                    combined_text = f"{row['original_question']} {' '.join(question_variations)}"
                    
                    # 임베딩 생성
                    embedding = self.embedding_service.get_embeddings(combined_text)[0]
                    
                    # 벡터 DB에 저장
                    self.vector_db_service.collection.add(
                        embeddings=[embedding],
                        documents=[combined_text],
                        metadatas=[{
                            "original_question": row['original_question'],
                            "structured_answer": row['structured_answer'],
                            "keywords": row['keywords']
                        }],
                        ids=[f"faq_{idx}"]
                    )
                    
                except Exception as e:
                    logger.error(f"FAQ 항목 처리 중 오류 발생 (인덱스 {idx}): {str(e)}")
                    continue
            
            logger.info("FAQ 데이터가 성공적으로 처리되었습니다.")
            return True
            
        except Exception as e:
            logger.error(f"FAQ 데이터 처리 중 오류 발생: {str(e)}")
            return False

    def find_faq_match(self, query: str, threshold: float = 0.75):
        """질문에 가장 잘 맞는 FAQ를 찾습니다."""
        try:
            if self.faq_data is None:
                logger.error("FAQ data not loaded")
                return None

            # 질문 임베딩 생성
            query_embedding = self.embedding_service.get_embeddings(query)[0]
            
            # 벡터 DB에서 유사한 문서 검색
            results = self.vector_db_service.query(query_embedding)
            
            if not results["ids"]:
                logger.warning("No matches found in vector database")
                return None

            # 가장 유사한 문서 찾기
            best_match_idx = 0
            best_match_distance = results["distances"][0][0]
            
            logger.info(f"Best match distance: {best_match_distance}")
            
            if best_match_distance > threshold:
                logger.warning(f"Best match distance ({best_match_distance}) too high")
                return None

            # 메타데이터에서 구조화된 답변 추출
            metadata = results["metadatas"][0][best_match_idx]
            structured_answer = json.loads(metadata["structured_answer"])
            
            # 답변 생성
            answer_parts = []
            
            # 기본 규칙 섹션
            if structured_answer.get('basic_rules'):
                answer_parts.append("[기본 규칙]\n")
                for rule in structured_answer['basic_rules']:
                    answer_parts.append(f"• {rule}\n")
            
            # 예시 섹션
            if structured_answer.get('examples'):
                answer_parts.append("\n[예시]\n")
                for example in structured_answer['examples']:
                    if isinstance(example, dict):
                        scenario = example.get('scenario', '')
                        result = example.get('result', '')
                        explanation = example.get('explanation', '')
                        formatted_example = f"• {scenario}\n  → {result}\n  (설명: {explanation})"
                    else:
                        formatted_example = f"• {example}"
                    answer_parts.append(f"{formatted_example}\n")
            
            # 주의사항 섹션
            if structured_answer.get('cautions'):
                answer_parts.append("\n[주의사항]\n")
                for caution in structured_answer['cautions']:
                    answer_parts.append(f"• {caution}\n")
            
            # 모든 섹션을 합쳐서 최종 답변 생성
            answer = "".join(answer_parts).strip()
            
            return {
                "answer": answer,
                "sources": ["FAQ 데이터베이스"],
                "is_faq": True,
                "match_distance": best_match_distance
            }
            
        except Exception as e:
            logger.error(f"Error finding FAQ match: {e}")
            return None 