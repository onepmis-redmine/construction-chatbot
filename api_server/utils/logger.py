import logging
from pathlib import Path
from datetime import datetime
import sys

class Logger:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.root_dir = Path(__file__).parent.parent.parent
        self.logs_dir = self.root_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # 로거 초기화
        self.setup_logging()
    
    def setup_logging(self):
        """로깅 설정을 초기화합니다."""
        # 로그 포맷터 설정
        log_formatter = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 로그 파일 핸들러 설정
        log_file = self.logs_dir / f"query_logs_{datetime.now().strftime('%Y%m')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(logging.INFO)
        
        # 콘솔 핸들러 설정
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_formatter)
        console_handler.setLevel(logging.INFO)
        console_handler.encoding = 'utf-8'
        
        # 루트 로거 설정
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # 기존 핸들러 제거
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 새 핸들러 추가
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # 시작 로그 메시지
        root_logger.info("로깅 시스템이 초기화되었습니다.")
        root_logger.info(f"로그 파일 경로: {log_file}")
    
    def get_logger(self, name=None):
        """지정된 이름의 로거를 반환합니다."""
        return logging.getLogger(name)

# 싱글톤 인스턴스 생성
logger = Logger()

def get_logger(name=None):
    """로거 인스턴스를 반환하는 편의 함수"""
    return logger.get_logger(name) 