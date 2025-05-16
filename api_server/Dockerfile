# 빌드 스테이지
FROM python:3.11-alpine as builder

WORKDIR /app

# 필요한 빌드 도구 설치
RUN apk add --no-cache \
    build-base \
    gcc \
    musl-dev \
    libffi-dev \
    openssl-dev

# 가상환경 생성 및 활성화
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    # 불필요한 파일 제거
    find /opt/venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type d -name "test" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type f -name "*.pyc" -delete && \
    find /opt/venv -type f -name "*.pyo" -delete && \
    find /opt/venv -type f -name "*.pyd" -delete && \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true && \
    # pip 캐시 정리
    pip cache purge && \
    # 문서 파일 제거
    find /opt/venv -type d -name "docs" -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type d -name "examples" -exec rm -rf {} + 2>/dev/null || true && \
    # 테스트 데이터 제거
    find /opt/venv -type d -name "datasets" -exec rm -rf {} + 2>/dev/null || true && \
    # 불필요한 모델 파일 제거
    rm -rf /root/.cache/huggingface && \
    # transformers 캐시 정리
    rm -rf /opt/venv/lib/python3.11/site-packages/transformers/models/* && \
    # torch 모듈 최적화
    rm -rf /opt/venv/lib/python3.11/site-packages/torch/test && \
    rm -rf /opt/venv/lib/python3.11/site-packages/torch/testing && \
    # 개발 관련 파일 제거
    find /opt/venv -type f -name "*.c" -delete && \
    find /opt/venv -type f -name "*.h" -delete && \
    find /opt/venv -type f -name "*.cpp" -delete && \
    find /opt/venv -type f -name "*.cc" -delete && \
    find /opt/venv -type f -name "*.hpp" -delete

# 최종 스테이지
FROM python:3.11-alpine as final

WORKDIR /app

# ud544uc694ud55c ub7f0ud0c0uc784 ud328ud0a4uc9c0ub9cc uc124uce58
RUN apk add --no-cache \
    libstdc++ \
    libgomp

# 가상환경 복사
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 필요한 디렉토리 생성
RUN mkdir -p /app/model_cache /app/vector_db /app/logs /app/docs && \
    # 권한 설정
    adduser -D appuser && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app

# 애플리케이션 파일 복사
COPY main.py .
COPY .env .
COPY utils/ ./utils/

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1 \
    TRANSFORMERS_CACHE=/app/model_cache \
    TORCH_HOME=/app/model_cache \
    # Render 환경 감지를 위한 환경 변수
    RENDER=true \
    # 추가 최적화를 위한 환경 변수
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# 비특권 사용자로 실행
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
