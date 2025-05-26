import httpx
import os
from utils.logger import get_logger

logger = get_logger(__name__)

class OpenRouterClient:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "openai/gpt-3.5-turbo"

    def call_openrouter(self, prompt: str, conversation_history=None) -> str:
        """OpenRouter API를 호출하여 응답을 받습니다."""
        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json"
        }
        
        # API 키가 없는 경우 기본 메시지 반환
        if not self.api_key or self.api_key == "your_openrouter_api_key_here":
            logger.error("OpenRouter API 키가 설정되지 않았습니다.")
            return "죄송합니다. OpenRouter API 키가 설정되지 않아 질문에 대한 답변을 제공할 수 없습니다. 관리자에게 문의해주세요."
        
        messages = []
        
        # 시스템 메시지 추가 (존댓말 지시)
        system_message = {
            "role": "system", 
            "content": "당신은 건설 정보 시스템에 대한 전문 지식을 갖춘 챗봇입니다. 항상 정중하고 공손한 존댓말로 답변해주세요."
        }
        messages.append(system_message)
        
        # 대화 기록이 있으면 추가
        if conversation_history:
            for msg in conversation_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # 현재 사용자 메시지 추가
        messages.append({"role": "user", "content": prompt})
        
        # 대화 기록이 없는 경우에도 시스템 메시지는 유지
        if len(messages) <= 2:  # 시스템 메시지 + 현재 메시지만 있는 경우
            messages = [system_message, {"role": "user", "content": prompt}]
        
        payload = {
            "model": self.model,
            "messages": messages
        }
        
        try:
            logger.info(f"OpenRouter API 호출: {prompt[:50]}...")
            
            response = httpx.post(
                url=self.base_url,
                headers=headers,
                json=payload,
                timeout=60.0,
                verify=False  # 테스트 환경에서만 사용
            )
            
            # 응답 상태 코드 로깅
            logger.info(f"OpenRouter API 응답 상태 코드: {response.status_code}")
            
            # 응답 내용 디버그를 위해 로깅 (개인정보는 제외)
            logger.info(f"OpenRouter API 응답 헤더: {dict(response.headers)}")
            
            if response.status_code != 200:
                error_msg = f"OpenRouter API 오류: 상태 코드 {response.status_code}"
                try:
                    error_data = response.json()
                    error_msg += f", 상세 내용: {error_data}"
                except:
                    error_msg += f", 응답: {response.text[:200]}"
                
                logger.error(error_msg)
                return f"죄송합니다. API 서버에 문제가 발생했습니다. 관리자에게 문의해주세요. (오류: {response.status_code})"
            
            data = response.json()
            
            # 응답 데이터 구조 확인
            if "choices" not in data or len(data["choices"]) == 0:
                logger.error(f"OpenRouter API 응답에 choices 필드가 없습니다: {data}")
                return "죄송합니다. API 응답이 올바르지 않습니다. 관리자에게 문의해주세요."
            
            if "message" not in data["choices"][0]:
                logger.error(f"OpenRouter API 응답에 message 필드가 없습니다: {data['choices'][0]}")
                return "죄송합니다. API 응답이 올바르지 않습니다. 관리자에게 문의해주세요."
            
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP 오류: {e.response.status_code} - {e.response.text}")
            return f"죄송합니다. API 서버 응답 오류가 발생했습니다. (상태 코드: {e.response.status_code})"
        except httpx.RequestError as e:
            logger.error(f"요청 오류: {e}")
            return "죄송합니다. 서버 연결에 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
        except Exception as e:
            logger.error(f"OpenRouter API 호출 중 오류 발생: {e}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
            return f"죄송합니다. 질문 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요. (오류: {str(e)[:100]})" 