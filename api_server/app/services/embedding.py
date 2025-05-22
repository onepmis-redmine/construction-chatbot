import torch
from transformers import AutoTokenizer, AutoModel
import gc
from utils.logger import get_logger

logger = get_logger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", cache_dir: str = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

    def load_model(self):
        """모델을 로드합니다."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                local_files_only=False
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                local_files_only=False
            ).to(self.device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"모델 로딩 중 오류 발생: {str(e)}")
            raise

    def unload_model(self):
        """모델을 메모리에서 해제합니다."""
        try:
            del self.model
            del self.tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded successfully")
        except Exception as e:
            logger.error(f"모델 언로딩 중 오류 발생: {str(e)}")
            raise

    def mean_pooling(self, model_output, attention_mask):
        """평균 풀링을 수행합니다."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 0) / torch.clamp(input_mask_expanded.sum(0), min=1e-9)

    def get_embeddings(self, texts, batch_size: int = 32):
        """텍스트의 임베딩을 생성합니다."""
        if not texts:
            logger.warning("Empty input texts")
            return []

        if isinstance(texts, str):
            texts = [texts]

        try:
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                encoded_input = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                ).to(self.device)

                with torch.no_grad():
                    model_output = self.model(**encoded_input)

                batch_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
                all_embeddings.extend(batch_embeddings.cpu().numpy())

            return all_embeddings
        except Exception as e:
            logger.error(f"임베딩 생성 중 오류 발생: {str(e)}")
            raise 