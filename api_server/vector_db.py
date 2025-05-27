import chromadb
import torch
from transformers import AutoTokenizer, AutoModel
import json
import logging
import gc
import numpy as np
from pathlib import Path
import asyncio
from utils.logger import get_logger
from typing import List, Union
import threading

logger = get_logger(__name__)

# ChromaDB 클라이언트 설정 - 전역 컬렉션 이름 상수 정의
COLLECTION_NAME = "construction_manuals"

class VectorDB:
    def __init__(self, vector_db_path: Path):
        self.vector_db_path = vector_db_path
        self.chroma_client = chromadb.PersistentClient(path=str(vector_db_path))
        self.collection = None
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = None
        self.model = None
        self._lock = threading.Lock()  # 스레드 안전을 위한 락 추가
        self._initialized = False

    def initialize(self):
        """모델과 토크나이저를 초기화합니다."""
        if not self._initialized:
            with self._lock:  # 초기화 시 락 사용
                if not self._initialized:  # Double-checked locking
                    try:
                        self.model = AutoModel.from_pretrained(self.model_name)
                        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                        self._initialized = True
                        logger.info("VectorDB 초기화 완료")
                    except Exception as e:
                        logger.error(f"VectorDB 초기화 중 오류 발생: {e}")
                        raise

    def get_collection(self):
        """현재 컬렉션을 가져오거나 없으면 생성합니다."""
        try:
            if self.collection is None:
                try:
                    # 기존 컬렉션 가져오기 시도
                    self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
                    logger.info(f"기존 컬렉션을 가져왔습니다: {COLLECTION_NAME}")
                except Exception as e:
                    logger.warning(f"기존 컬렉션을 가져오는데 실패했습니다: {e}")
                    # 컬렉션 생성
                    self.collection = self.chroma_client.create_collection(name=COLLECTION_NAME)
                    logger.info(f"새 컬렉션을 생성했습니다: {COLLECTION_NAME}")
            
            # 컬렉션 유효성 검사
            dummy_result = self.collection.count()
            logger.info(f"컬렉션 항목 수: {dummy_result}")
            return self.collection
        except Exception as e:
            logger.error(f"컬렉션 가져오기/생성 중 오류: {e}")
            # 마지막 시도: 모든 컬렉션 삭제 후 새로 생성
            try:
                self.chroma_client.delete_collection(name=COLLECTION_NAME)
                self.collection = self.chroma_client.create_collection(name=COLLECTION_NAME)
                logger.info(f"컬렉션을 강제로 재생성했습니다: {COLLECTION_NAME}")
                return self.collection
            except Exception as e2:
                logger.error(f"컬렉션 강제 재생성 중 오류: {e2}")
                raise e2

    def load_model(self):
        """모델을 로드합니다."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=False)
        if self.model is None:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                local_files_only=False
            )
            self.model.eval()  # 평가 모드로 설정
        return self.tokenizer, self.model

    def unload_model(self):
        """모델을 언로드하고 메모리를 정리합니다."""
        del self.tokenizer
        del self.model
        self.tokenizer = None
        self.model = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()

    def mean_pooling(self, model_output, attention_mask):
        """Mean Pooling 함수"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """텍스트의 임베딩을 생성합니다."""
        if not self._initialized:
            self.initialize()

        if isinstance(texts, str):
            texts = [texts]

        logger.info(f"get_embeddings: 처리할 텍스트 {len(texts)}개")

        try:
            with self._lock:  # 임베딩 생성 시 락 사용
                # 토크나이저가 None인 경우 재초기화
                if self.tokenizer is None:
                    self.initialize()
                
                encoded_input = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                )

                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                    embeddings = model_output.last_hidden_state.mean(dim=1)
                    embeddings = embeddings.numpy().tolist()

                logger.info(f"생성된 임베딩 형식: 차원 수 = {len(embeddings)}, 첫 임베딩 길이 = {len(embeddings[0])}")
                return embeddings

        except Exception as e:
            logger.error(f"임베딩 생성 중 오류 발생: {e}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
            # 오류 발생 시 재초기화 시도
            self._initialized = False
            self.initialize()
            return [[0.0] * 384] * len(texts)  # 기본값 반환

    async def initialize_vector_db(self, faq_data, force_rebuild=False):
        """구조화된 FAQ 데이터를 벡터 데이터베이스에 임베딩합니다."""
        try:
            # 기존 컬렉션이 있는지 확인
            existing_collection = False
            try:
                test_collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
                collection_count = test_collection.count()
                if collection_count > 0:
                    existing_collection = True
                    logger.info(f"기존 컬렉션이 존재함: {COLLECTION_NAME}, 항목 수: {collection_count}")
            except Exception as e:
                logger.warning(f"기존 컬렉션 확인 중 오류 발생: {e}")
                existing_collection = False
            
            # 기존 컬렉션이 있고 강제 재구축이 아니면 그대로 사용
            if existing_collection and not force_rebuild:
                self.collection = test_collection
                logger.info(f"기존 컬렉션을 재사용합니다: {COLLECTION_NAME}")
                return {"success": True, "message": f"기존 컬렉션 재사용 ({collection_count}개 항목)"}
            
            if faq_data is None or faq_data.empty:
                logger.error("FAQ 데이터가 비어 있습니다.")
                return {"success": False, "message": "FAQ 데이터가 비어 있습니다."}
            
            logger.info("새 벡터 데이터베이스를 생성합니다...")
            
            # 기존 컬렉션 삭제 후 재생성
            try:
                self.chroma_client.delete_collection(name=COLLECTION_NAME)
                logger.info(f"기존 컬렉션 삭제 완료: {COLLECTION_NAME}")
            except Exception as e:
                logger.warning(f"컬렉션 삭제 중 오류 발생 (무시됨): {e}")
                
            # 새 컬렉션 생성
            self.collection = self.chroma_client.create_collection(name=COLLECTION_NAME)
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
                
                # 배치 크기 줄이기
                batch_size = 10
                
                # embeddings 리스트 초기화
                embeddings = []
                
                # 메모리 사용량 최적화
                for i in range(0, len(texts), batch_size):
                    batch_end = min(i + batch_size, len(texts))
                    batch = texts[i:batch_end]
                    
                    logger.info(f"임베딩 생성 진행률: {batch_end / len(texts) * 100:.1f}% ({batch_end}/{len(texts)})")
                    
                    # 배치 단위로 임베딩 생성
                    batch_embeddings = self.get_embeddings(batch)
                    
                    # 임베딩 형식 확인 및 수정
                    if batch_embeddings:
                        while isinstance(batch_embeddings, list) and len(batch_embeddings) > 0 and isinstance(batch_embeddings[0], list) and isinstance(batch_embeddings[0][0], list):
                            batch_embeddings = [emb for sublist in batch_embeddings for emb in sublist]
                            logger.info("3차원 임베딩을 2차원으로 변환했습니다.")

                        if batch_embeddings and isinstance(batch_embeddings[0], (float, int)):
                            batch_embeddings = [batch_embeddings]
                            logger.info("단일 임베딩을 리스트로 변환했습니다.")

                        embeddings.extend(batch_embeddings)
                    
                    # 메모리 정리
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # 잠시 대기하여 메모리 해제 시간 확보
                    await asyncio.sleep(0.1)
                
                logger.info("임베딩 생성 완료! 이제 ChromaDB에 저장합니다...")
                
                # 임베딩 형식 로깅 (디버깅용)
                if embeddings:
                    logger.info(f"첫 번째 임베딩 차원 구조: {type(embeddings)}, {type(embeddings[0])}")
                    if isinstance(embeddings[0], list):
                        logger.info(f"첫 번째 임베딩 길이: {len(embeddings[0])}")
                
                # 최종 임베딩 형식 검증
                if not all(isinstance(emb, list) and all(isinstance(x, (int, float)) for x in emb) for emb in embeddings):
                    logger.error("임베딩 형식이 올바르지 않습니다.")
                    return {"success": False, "message": "임베딩 형식이 올바르지 않습니다."}
                
                # === 임베딩 차원 강제 변환 (최종 방어) ===
                if isinstance(embeddings, np.ndarray):
                    embeddings = embeddings.tolist()
                while isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], list) and isinstance(embeddings[0][0], list):
                    embeddings = [emb for sublist in embeddings for emb in sublist]
                if embeddings and isinstance(embeddings[0], (float, int)):
                    embeddings = [embeddings]
                # === 방어 끝 ===

                # 컬렉션에 추가
                self.collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                
                logger.info(f"벡터 데이터베이스에 {len(texts)}개 항목 추가 완료")
                
                # 간단한 테스트 쿼리 실행
                test_emb = self.get_embeddings("테스트 쿼리")
                while isinstance(test_emb, list) and len(test_emb) > 0 and isinstance(test_emb[0], list) and isinstance(test_emb[0][0], list):
                    test_emb = [emb for sublist in test_emb for emb in sublist]
                if test_emb and isinstance(test_emb[0], (float, int)):
                    test_emb = [test_emb]
                test_result = self.collection.query(
                    query_embeddings=test_emb,
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