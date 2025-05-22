import re
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """텍스트를 정리합니다."""
    if not text:
        return ""
    
    # HTML 태그 제거
    text = re.sub(r'<[^>]+>', '', text)
    
    # 특수문자 제거
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    
    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def format_faq_answer(match: Dict[str, Any]) -> str:
    """FAQ 답변을 포맷팅합니다."""
    answer_parts = []
    
    # 기본 규칙
    if match.get("basic_rules"):
        answer_parts.append("📌 기본 규칙")
        answer_parts.append(match["basic_rules"])
    
    # 예시
    if match.get("examples"):
        answer_parts.append("\n💡 예시")
        answer_parts.append(match["examples"])
    
    # 주의사항
    if match.get("cautions"):
        answer_parts.append("\n⚠️ 주의사항")
        answer_parts.append(match["cautions"])
    
    return "\n".join(answer_parts)

def extract_keywords(text: str) -> List[str]:
    """텍스트에서 키워드를 추출합니다."""
    if not text:
        return []
    
    # 불용어 목록
    stop_words = {"이", "그", "저", "것", "등", "및", "또는", "그리고"}
    
    # 단어 추출 및 정리
    words = re.findall(r'\w+', text.lower())
    keywords = [word for word in words if word not in stop_words and len(word) > 1]
    
    return list(set(keywords)) 