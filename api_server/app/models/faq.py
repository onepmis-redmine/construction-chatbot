from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Question(BaseModel):
    query: str
    session_id: Optional[str] = None

class FAQResponse(BaseModel):
    answer: str
    sources: List[str]
    is_faq: bool
    error: Optional[str] = None
    match_distance: Optional[float] = None 