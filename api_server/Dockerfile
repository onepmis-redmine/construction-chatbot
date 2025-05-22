# 빌드 스테이지
FROM python:3.11-slim as builder

WORKDIR /build

# 필요한 빌드 도구만 설치하고 캐시 정리
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 가상환경 생성 및 활성화
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 의존성 파일 복사 및 설치
COPY api_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip/*

# 모델 미리 다운로드
RUN mkdir -p /build/model_cache && \
    python -c "from transformers import AutoTokenizer, AutoModel; \
    AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', cache_dir='/build/model_cache'); \
    AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', cache_dir='/build/model_cache')"

# 최종 스테이지
FROM python:3.11-slim as final

WORKDIR /app

# 가상환경 복사
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 모델 캐시 복사
COPY --from=builder /build/model_cache /model_cache

# 필요한 디렉토리 생성 및 권한 설정
RUN mkdir -p /vector_db /logs /docs /sessions /questions /frontend/build && \
    adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app /model_cache /vector_db /logs /docs /sessions /questions /frontend && \
    chmod -R 755 /app /model_cache /vector_db /logs /docs /sessions /questions /frontend

# 애플리케이션 파일 복사
COPY api_server/ .

# FAQ 파일 복사
COPY docs/enhanced_qa_pairs.xlsx /docs/
RUN chown -R appuser:appuser /docs

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/model_cache \
    TRANSFORMERS_CACHE=/model_cache \
    TORCH_HOME=/model_cache \
    # Render 환경 감지를 위한 환경 변수
    RENDER=true \
    # 추가 최적화를 위한 환경 변수
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # OpenRouter API 키
    OPENROUTER_API_KEY=${OPENROUTER_API_KEY} \
    # 메모리 최적화를 위한 환경 변수
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

# 비특권 사용자로 실행
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
