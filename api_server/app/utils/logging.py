import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

from app.config import LOGS_DIR

def setup_logger(name: str) -> logging.Logger:
    """로거 설정을 초기화합니다."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 로그 포맷 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러
    log_file = LOGS_DIR / f"{name}.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger 