import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
import os

def get_logger(name: str) -> logging.Logger:
    """로거를 생성하고 설정합니다."""
    # 로거 생성
    logger = logging.getLogger(name)
    
    # 이미 핸들러가 설정되어 있다면 그대로 반환
    if logger.handlers:
        return logger
    
    # 로그 레벨 설정
    logger.setLevel(logging.INFO)
    
    # 로그 포맷 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 설정
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "app.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger 