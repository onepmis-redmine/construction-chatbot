import chromadb
from pathlib import Path
import os
from utils.logger import get_logger

logger = get_logger(__name__)

class VectorDBService:
    def __init__(self, vector_db_path: Path, collection_name: str = "construction_manuals"):
        # 경로를 절대 경로로 변환하고 디렉토리 생성
        self.vector_db_path = Path(vector_db_path).resolve()
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        logger.info(f"Vector DB 경로: {self.vector_db_path}")

    def initialize(self):
        """벡터 DB 클라이언트와 컬렉션을 초기화합니다."""
        try:
            # 클라이언트 초기화
            self.client = chromadb.PersistentClient(path=str(self.vector_db_path))
            logger.info("ChromaDB 클라이언트가 초기화되었습니다.")
            
            # 컬렉션 초기화
            self._get_or_create_collection()
            
            # 초기화 확인
            count = self.collection.count()
            logger.info(f"벡터 DB 초기화 완료. 현재 컬렉션 항목 수: {count}")
            return True
        except Exception as e:
            logger.error(f"벡터 DB 초기화 중 오류 발생: {str(e)}")
            return False

    def _get_or_create_collection(self):
        """컬렉션을 가져오거나 생성합니다."""
        try:
            if self.collection is None:
                try:
                    self.collection = self.client.get_collection(name=self.collection_name)
                    logger.info(f"기존 컬렉션을 가져왔습니다: {self.collection_name}")
                except Exception as e:
                    logger.warning(f"기존 컬렉션을 가져오는데 실패했습니다: {e}")
                    self.collection = self.client.create_collection(name=self.collection_name)
                    logger.info(f"새 컬렉션을 생성했습니다: {self.collection_name}")
            
            # 컬렉션 유효성 검사
            count = self.collection.count()
            logger.info(f"컬렉션 항목 수: {count}")
            return self.collection
        except Exception as e:
            logger.error(f"컬렉션 가져오기/생성 중 오류: {e}")
            # 마지막 시도: 모든 컬렉션 삭제 후 새로 생성
            try:
                self.client.delete_collection(name=self.collection_name)
                self.collection = self.client.create_collection(name=self.collection_name)
                logger.info(f"컬렉션을 강제로 재생성했습니다: {self.collection_name}")
                return self.collection
            except Exception as e2:
                logger.error(f"컬렉션 강제 재생성 중 오류: {e2}")
                raise e2

    def add_documents(self, embeddings, documents, metadatas, ids):
        """문서를 벡터 DB에 추가합니다."""
        try:
            if self.collection is None:
                self._get_or_create_collection()
            
            # 입력 데이터 검증
            if not all(len(x) == len(embeddings) for x in [documents, metadatas, ids]):
                raise ValueError("모든 입력 리스트의 길이가 일치해야 합니다.")
            
            # 문서 추가
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            # 추가 확인
            new_count = self.collection.count()
            logger.info(f"문서 추가 완료. 현재 컬렉션 항목 수: {new_count}")
            return True
        except Exception as e:
            logger.error(f"문서 추가 중 오류 발생: {str(e)}")
            return False

    def query(self, query_embedding, n_results=5):
        """벡터 DB에서 유사한 문서를 검색합니다."""
        try:
            if self.collection is None:
                self._get_or_create_collection()
            
            # 컬렉션 항목 수 확인
            count = self.collection.count()
            if count == 0:
                logger.warning("벡터 DB가 비어있습니다.")
                return {"ids": [], "distances": [], "metadatas": []}
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, count),
                include=["metadatas", "distances"]
            )
            
            if not results["ids"]:
                logger.warning("검색 결과가 없습니다.")
            
            return results
        except Exception as e:
            logger.error(f"벡터 DB 쿼리 중 오류 발생: {str(e)}")
            return {"ids": [], "distances": [], "metadatas": []} 