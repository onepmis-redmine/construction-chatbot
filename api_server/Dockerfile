# 빌드 스테이지
FROM python:3.11-slim as builder

WORKDIR /build

# 필요한 빌드 도구 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 가상환경 생성 및 활성화
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 의존성 파일 복사 및 설치
COPY api_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 최종 스테이지
FROM python:3.11-slim as final

WORKDIR /

# 가상환경 복사
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 필요한 디렉토리 생성 및 권한 설정
RUN mkdir -p /model_cache /vector_db /logs /docs /sessions /questions && \
    adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /model_cache /vector_db /logs /docs /sessions /questions && \
    chmod -R 755 /model_cache /vector_db /logs /docs /sessions /questions

# 애플리케이션 파일 복사
COPY api_server /app

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/model_cache \
    TORCH_HOME=/model_cache \
    # Render 환경 감지를 위한 환경 변수
    RENDER=true \
    # 추가 최적화를 위한 환경 변수
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# 비특권 사용자로 실행
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
