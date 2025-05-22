import torch
from transformers import AutoTokenizer, AutoModel
import gc
from utils.logger import get_logger

logger = get_logger(__name__)

class EmbeddingService:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load_model(self):
        try:
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    local_files_only=True,
                    cache_dir="model_cache"
                )
            if self.model is None:
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    local_files_only=True,
                    cache_dir="model_cache"
                )
                self.model.eval()
            return self.tokenizer, self.model
        except Exception as e:
            logger.error(f"모델 로딩 중 오류 발생: {str(e)}")
            raise

    def unload_model(self):
        del self.tokenizer
        del self.model
        self.tokenizer = None
        self.model = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, texts, batch_size=32):
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            logger.warning("get_embeddings: 입력 텍스트가 비어 있습니다.")
            return []
            
        logger.info(f"get_embeddings: 처리할 텍스트 {len(texts)}개")
        
        if len(texts) > batch_size:
            logger.info(f"배치 크기({len(texts)})가 제한({batch_size})을 초과하여 분할 처리합니다.")
            result = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.get_embeddings(batch_texts)
                result.extend(batch_embeddings)
            return result
        
        try:
            tokenizer, model = self.load_model()
            
            encoded_input = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                model_output = model(**encoded_input)
            
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            
            return sentence_embeddings.tolist()
            
        except Exception as e:
            logger.error(f"임베딩 생성 중 오류 발생: {e}")
            raise 