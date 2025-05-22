import chromadb
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)

class VectorDBService:
    def __init__(self, vector_db_path: Path, collection_name: str = "construction_manuals"):
        self.vector_db_path = vector_db_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None

    def initialize(self):
        """벡터 DB 클라이언트와 컬렉션을 초기화합니다."""
        try:
            self.client = chromadb.PersistentClient(path=str(self.vector_db_path))
            self._get_or_create_collection()
            return True
        except Exception as e:
            logger.error(f"벡터 DB 초기화 중 오류 발생: {e}")
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
            dummy_result = self.collection.count()
            logger.info(f"컬렉션 항목 수: {dummy_result}")
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

    def query(self, query_embedding, n_results=5):
        """벡터 DB에서 유사한 문서를 검색합니다."""
        try:
            if self.collection is None:
                self._get_or_create_collection()
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["metadatas", "distances"]
            )
            return results
        except Exception as e:
            logger.error(f"벡터 DB 쿼리 중 오류 발생: {e}")
            raise 