from datetime import datetime, timedelta
import uuid
from typing import Dict, List, Optional
import json
from pathlib import Path
import os
from utils.logger import get_logger

logger = get_logger(__name__)

class ChatSession:
    def __init__(self, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.messages: List[Dict] = []
        
    def add_message(self, role: str, content: str):
        """대화 메시지를 추가합니다."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.messages.append(message)
        self.last_activity = datetime.now()
        
    def get_messages(self) -> List[Dict]:
        """세션의 모든 메시지를 반환합니다."""
        return self.messages
    
    def to_dict(self) -> Dict:
        """세션을 딕셔너리로 변환합니다."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "messages": self.messages
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ChatSession':
        """딕셔너리에서 세션을 생성합니다."""
        session = cls(data["session_id"])
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.last_activity = datetime.fromisoformat(data["last_activity"])
        session.messages = data["messages"]
        return session

class SessionManager:
    def __init__(self, sessions_dir: Path):
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.session_timeout = timedelta(hours=24)  # 24시간 후 세션 만료

    def _get_session_file(self, session_id: str) -> Path:
        """세션 파일 경로를 반환합니다."""
        return self.sessions_dir / f"{session_id}.json"

    def create_session(self, session_id: str) -> Dict:
        """새로운 세션을 생성합니다."""
        session_data = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "conversation_history": []
        }
        
        session_file = self._get_session_file(session_id)
        try:
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"새로운 세션 생성: {session_id}")
            return session_data
        except Exception as e:
            logger.error(f"세션 생성 중 오류 발생: {e}")
            return session_data

    def get_session(self, session_id: str) -> Optional[Dict]:
        """세션 데이터를 가져옵니다."""
        session_file = self._get_session_file(session_id)
        
        if not session_file.exists():
            logger.info(f"세션 파일이 없음: {session_id}")
            return self.create_session(session_id)
        
        try:
            with open(session_file, "r", encoding="utf-8") as f:
                session_data = json.load(f)
            
            # 세션 만료 확인
            last_activity = datetime.fromisoformat(session_data["last_activity"])
            if datetime.now() - last_activity > self.session_timeout:
                logger.info(f"세션 만료: {session_id}")
                self.delete_session(session_id)
                return self.create_session(session_id)
            
            # conversation_history가 없으면 초기화
            if "conversation_history" not in session_data:
                session_data["conversation_history"] = []
                logger.info(f"대화 기록 초기화: {session_id}")
            
            return session_data
        except Exception as e:
            logger.error(f"세션 데이터 로드 중 오류 발생: {e}")
            return self.create_session(session_id)

    def update_session(self, session_id: str, message: Dict) -> Optional[Dict]:
        """세션 데이터를 업데이트합니다."""
        session_data = self.get_session(session_id)
        if not session_data:
            return None
        
        session_data["last_activity"] = datetime.now().isoformat()
        session_data["conversation_history"].append(message)
        
        # 대화 기록이 너무 길어지면 오래된 메시지 제거
        if len(session_data["conversation_history"]) > 50:
            session_data["conversation_history"] = session_data["conversation_history"][-50:]
        
        try:
            session_file = self._get_session_file(session_id)
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            
            return session_data
        except Exception as e:
            logger.error(f"세션 데이터 저장 중 오류 발생: {e}")
            return None

    def delete_session(self, session_id: str) -> bool:
        """세션을 삭제합니다."""
        try:
            session_file = self._get_session_file(session_id)
            if session_file.exists():
                session_file.unlink()
                logger.info(f"세션 삭제 완료: {session_id}")
            return True
        except Exception as e:
            logger.error(f"세션 삭제 중 오류 발생: {e}")
            return False

    def cleanup_expired_sessions(self) -> int:
        """만료된 세션을 정리합니다."""
        count = 0
        for session_file in self.sessions_dir.glob("*.json"):
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    session_data = json.load(f)
                
                last_activity = datetime.fromisoformat(session_data["last_activity"])
                if datetime.now() - last_activity > self.session_timeout:
                    session_file.unlink()
                    count += 1
                    logger.info(f"만료된 세션 정리: {session_file.stem}")
            except Exception as e:
                logger.error(f"세션 정리 중 오류 발생 ({session_file.name}): {e}")
        
        return count 