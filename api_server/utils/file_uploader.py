import os
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict
from fastapi import UploadFile, HTTPException
from datetime import datetime
from utils.logger import get_logger

logger = get_logger(__name__)

class FileUploader:
    def __init__(self, upload_dir: Path, docs_dir: Path):
        self.upload_dir = upload_dir
        self.docs_dir = docs_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)

    async def save_upload_file(self, file: UploadFile) -> Tuple[bool, str]:
        """업로드된 파일을 저장합니다."""
        try:
            # 파일 확장자 검사
            if not file.filename.endswith('.xlsx'):
                return False, "Excel 파일(.xlsx)만 업로드 가능합니다."

            # 파일 저장 경로 설정
            file_path = self.upload_dir / file.filename
            
            # 파일 저장
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            logger.info(f"파일 업로드 완료: {file.filename}")
            return True, str(file_path)
        except Exception as e:
            logger.error(f"파일 업로드 중 오류 발생: {e}")
            return False, f"파일 업로드 중 오류가 발생했습니다: {str(e)}"

    def get_enhanced_faq_path(self) -> Optional[Path]:
        """구조화된 FAQ 파일 경로를 반환합니다."""
        faq_path = self.docs_dir / "enhanced_qa_pairs.xlsx"
        return faq_path if faq_path.exists() else None

    def cleanup_old_files(self, max_age_days: int = 7) -> int:
        """오래된 파일을 정리합니다."""
        count = 0
        try:
            current_time = datetime.now()
            for file_path in self.upload_dir.glob("*"):
                if file_path.is_file():
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    file_age = (current_time - file_mtime).days
                    if file_age > max_age_days:
                        file_path.unlink()
                        count += 1
                        logger.info(f"오래된 파일 삭제: {file_path.name}")
        except Exception as e:
            logger.error(f"파일 정리 중 오류 발생: {e}")
        return count

    def get_file_size(self, file_path: Path) -> Optional[int]:
        """파일 크기를 반환합니다."""
        try:
            return file_path.stat().st_size if file_path.exists() else None
        except Exception as e:
            logger.error(f"파일 크기 확인 중 오류 발생: {e}")
            return None

    def is_file_valid(self, file_path: Path) -> bool:
        """파일이 유효한지 확인합니다."""
        try:
            return file_path.exists() and file_path.is_file() and file_path.stat().st_size > 0
        except Exception as e:
            logger.error(f"파일 유효성 검사 중 오류 발생: {e}")
            return False 