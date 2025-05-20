from datetime import datetime
import uuid
from typing import Dict, List, Optional
import json
from pathlib import Path
import os

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
    def __init__(self, storage_dir: str = "sessions"):
        self.sessions: Dict[str, ChatSession] = {}
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._load_sessions()
        
    def _load_sessions(self):
        """저장된 세션들을 로드합니다."""
        for file_path in self.storage_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    session = ChatSession.from_dict(data)
                    self.sessions[session.session_id] = session
            except Exception as e:
                print(f"세션 로드 중 오류 발생: {e}")
                
    def _save_session(self, session: ChatSession):
        """세션을 파일로 저장합니다."""
        file_path = self.storage_dir / f"{session.session_id}.json"
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"세션 저장 중 오류 발생: {e}")
            
    def create_session(self) -> ChatSession:
        """새로운 세션을 생성합니다."""
        session = ChatSession()
        self.sessions[session.session_id] = session
        self._save_session(session)
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """세션 ID로 세션을 가져옵니다."""
        return self.sessions.get(session_id)
    
    def add_message(self, session_id: str, role: str, content: str):
        """세션에 메시지를 추가합니다."""
        session = self.get_session(session_id)
        if session:
            session.add_message(role, content)
            self._save_session(session)
            
    def get_messages(self, session_id: str) -> List[Dict]:
        """세션의 모든 메시지를 가져옵니다."""
        session = self.get_session(session_id)
        return session.get_messages() if session else []
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """오래된 세션을 정리합니다."""
        current_time = datetime.now()
        sessions_to_remove = []
        
        for session_id, session in self.sessions.items():
            age = current_time - session.last_activity
            if age.total_seconds() > max_age_hours * 3600:
                sessions_to_remove.append(session_id)
                
        for session_id in sessions_to_remove:
            session = self.sessions.pop(session_id)
            file_path = self.storage_dir / f"{session_id}.json"
            try:
                file_path.unlink()
            except Exception as e:
                print(f"세션 파일 삭제 중 오류 발생: {e}") 