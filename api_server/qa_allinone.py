import os
import json
import pandas as pd
from pathlib import Path
import logging
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm
import sys
import codecs
from typing import Dict
import traceback  # 추가: 상세 에러 추적을 위해

# Windows 콘솔 출력 인코딩 설정
if sys.platform.startswith('win'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 프로젝트 루트 디렉토리 설정
ROOT_DIR = Path(__file__).parent.parent
DOCS_DIR = ROOT_DIR / "docs"
EXCEL_PATH = DOCS_DIR / "qa_pairs.xlsx"
ENHANCED_FAQ_PATH = DOCS_DIR / "enhanced_qa_pairs.xlsx"
LOG_DIR = ROOT_DIR / "logs"

# 경로 디버깅 로그 추가
print(f"현재 스크립트 경로: {Path(__file__)}")
print(f"루트 디렉토리: {ROOT_DIR}")
print(f"DOCS 디렉토리: {DOCS_DIR}")
print(f"QA 엑셀 파일 경로: {EXCEL_PATH}")
print(f"향상된 QA 파일 경로: {ENHANCED_FAQ_PATH}")

# 필요한 디렉토리 생성
DOCS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 환경 변수 로드
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    print("ERROR: GOOGLE_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")
    print("현재 .env 파일 위치:", ROOT_DIR / ".env")
    sys.exit(1)

try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"ERROR: Google API 설정 중 오류 발생: {str(e)}")
    print("상세 에러:", traceback.format_exc())
    sys.exit(1)

# 로깅 설정
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_file_handler = logging.FileHandler(LOG_DIR / 'qa_enhancement.log', encoding='utf-8')
log_file_handler.setFormatter(log_formatter)
log_stream_handler = logging.StreamHandler(sys.stdout)
log_stream_handler.setFormatter(log_formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(log_file_handler)
logger.addHandler(log_stream_handler)

class QAProcessor:
    def __init__(self):
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
            logger.info("QAProcessor initialized successfully")
            logger.info(f"EXCEL_PATH 확인: {EXCEL_PATH}, 존재여부: {EXCEL_PATH.exists()}")
        except Exception as e:
            print(f"ERROR: QAProcessor 초기화 중 오류 발생: {str(e)}")
            print("상세 에러:", traceback.format_exc())
            sys.exit(1)

    def read_qa_excel(self) -> pd.DataFrame:
        """FAQ 엑셀 파일을 읽어옵니다."""
        try:
            if not EXCEL_PATH.exists():
                error_msg = f"FAQ 엑셀 파일이 없습니다: {EXCEL_PATH}"
                logger.error(error_msg)
                print(f"ERROR: {error_msg}")
                raise FileNotFoundError(error_msg)
                
            df = pd.read_excel(EXCEL_PATH)
            logger.info(f"Successfully read FAQ file: {EXCEL_PATH}")
            return df
        except Exception as e:
            error_msg = f"Error reading FAQ file: {str(e)}"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            print("상세 에러:", traceback.format_exc())
            raise

    def enhance_qa_pair(self, question: str, answer: str) -> Dict:
        """Gemini를 사용하여 질문-답변 쌍을 구조화합니다."""
        prompt = f"""
다음 질문-답변 쌍을 구조화된 형식으로 변환해주세요.
결과는 반드시 올바른 JSON 형식이어야 합니다.
모든 답변은 반드시 정중하고 공손한 존댓말로 작성해주세요.
질문 변형도 가능한 존댓말로 작성하되, 다양한 표현 방식을 포함해주세요.

원본 질문: {question}
원본 답변: {answer}

다음 구조로 JSON을 생성해주세요:
{{
    "question_variations": [
        "변형된 질문 1 (존댓말로 작성)",
        "변형된 질문 2 (존댓말로 작성)",
        "변형된 질문 3 (존댓말로 작성)"
    ],
    "structured_answer": {{
        "basic_rules": [
            "기본 규칙/조건 1 (존댓말로 작성)",
            "기본 규칙/조건 2 (존댓말로 작성)"
        ],
        "examples": [
            {{
                "scenario": "예시 상황 설명 (존댓말로 작성)",
                "result_a": "A의 결과 (존댓말로 작성)",
                "result_b": "B의 결과 (존댓말로 작성)"
            }},
            {{
                "scenario": "다른 예시 상황 (존댓말로 작성)",
                "result_c": "C의 결과 (존댓말로 작성)"
            }}
        ],
        "cautions": [
            "주의사항 1 (존댓말로 작성)",
            "주의사항 2 (존댓말로 작성)"
        ]
    }},
    "keywords": [
        "키워드1",
        "키워드2",
        "키워드3",
        "키워드4",
        "키워드5",
        "키워드6"
    ]
}}

특별 지시사항:
1. keywords 항목에는 최소 6개 이상의 키워드를 포함해주세요.
2. 한글에 포함된 한자어와 그 순우리말 동의어를 모두 키워드에 포함해주세요. (예: '근로자'와 '일꾼', '휴일'과 '쉬는 날')
3. 띄어쓰기 유무에 따른 변형도 별도 키워드로 포함해주세요. (예: '주휴수당'과 '주휴 수당')
4. 유사 개념이지만 다른 의미의 키워드도 포함해주세요. (예: '주휴수당'과 '휴일수당'은 구분)
5. 실무에서 흔히 쓰이는 줄임말이나 약어도 포함해주세요. (예: '퇴직금'과 '퇴직금여')
6. 질문에 포함된 핵심 단어는 반드시 모두 키워드에 포함해주세요.

응답은 반드시 위와 같은 형식의 유효한 JSON이어야 합니다.
모든 설명과 답변은 반드시 정중하고 공손한 존댓말로 작성해주세요.
특히 주휴수당 관련 예시의 경우, A와 B는 한 예시에서 함께 설명하고, C는 별도의 예시로 구분해주세요.
"""
        try:
            response = self.model.generate_content(prompt)
            json_str = response.text
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].strip()
            
            enhanced_content = json.loads(json_str)
            logger.info(f"Successfully enhanced Q&A pair: {question}")
            return enhanced_content
        except Exception as e:
            error_msg = f"Error enhancing Q&A pair: {str(e)}"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            print("상세 에러:", traceback.format_exc())
            return None

    def process_and_enhance_qa(self):
        """FAQ를 읽고 구조화합니다."""
        try:
            df = self.read_qa_excel()
            enhanced_data = []

            for idx, row in tqdm(df.iterrows(), total=len(df), desc="FAQ 구조화 중"):
                question = str(row['질문']).strip()
                answer = str(row['답변']).strip()
                
                enhanced = self.enhance_qa_pair(question, answer)
                if enhanced:
                    # 데이터 검증
                    required_fields = ['question_variations', 'structured_answer', 'keywords']
                    if all(field in enhanced for field in required_fields):
                        # JSON 필드를 문자열로 변환
                        enhanced = {
                            'original_question': question,
                            'question_variations': json.dumps(enhanced['question_variations'], ensure_ascii=False, indent=2),
                            'structured_answer': json.dumps(enhanced['structured_answer'], ensure_ascii=False, indent=2),
                            'keywords': json.dumps(enhanced['keywords'], ensure_ascii=False, indent=2)
                        }
                        enhanced_data.append(enhanced)
                        print(f"[구조화 결과] #{idx+1} 질문: {question}")
                        print(f"[구조화 결과] 변형 질문: {enhanced['question_variations']}")
                        print(f"[구조화 결과] 구조화 답변: {enhanced['structured_answer']}")
                        print(f"[구조화 결과] 키워드: {enhanced['keywords']}")
                        print(f"Successfully processed Q&A pair {idx + 1}: {question}")
                    else:
                        error_msg = f"Missing required fields in enhanced data for question: {question}"
                        logger.error(error_msg)
                        print(f"ERROR: {error_msg}")
                logger.info(f"Processed {idx + 1}/{len(df)} QA pairs")

            if not enhanced_data:
                error_msg = "No FAQ data was successfully processed"
                logger.error(error_msg)
                print(f"ERROR: {error_msg}")
                return []

            # 결과를 새로운 엑셀 파일로 저장
            output_df = pd.DataFrame(enhanced_data)
            
            # 저장 전 데이터 확인
            logger.debug(f"Enhanced data before saving:\n{output_df.to_string()}")
            
            try:
                output_df.to_excel(ENHANCED_FAQ_PATH, index=False)
                logger.info(f"Successfully saved enhanced FAQ to: {ENHANCED_FAQ_PATH}")
            except Exception as e:
                error_msg = f"Error saving enhanced FAQ: {str(e)}"
                logger.error(error_msg)
                print(f"ERROR: {error_msg}")
                print("상세 에러:", traceback.format_exc())
                
            return enhanced_data
        except Exception as e:
            error_msg = f"Error in process_and_enhance_qa: {str(e)}"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            print("상세 에러:", traceback.format_exc())
            return []

def process_qa_file():
    """FAQ 파일을 처리하고 구조화합니다."""
    try:
        processor = QAProcessor()
        enhanced_data = processor.process_and_enhance_qa()
        logger.info(f"FAQ 구조화 완료. 총 {len(enhanced_data)}개의 FAQ가 처리되었습니다.")
        return enhanced_data
    except Exception as e:
        error_msg = f"Error in process_qa_file: {str(e)}"
        logger.error(error_msg)
        print(f"ERROR: {error_msg}")
        print("상세 에러:", traceback.format_exc())
        return []

def main():
    """메인 함수"""
    try:
        process_qa_file()
        logger.info("FAQ 구조화 처리가 완료되었습니다.")
        return True
    except Exception as e:
        error_msg = f"FAQ 구조화 처리 중 오류 발생: {str(e)}"
        logger.error(error_msg)
        print(f"ERROR: {error_msg}")
        print("상세 에러:", traceback.format_exc())
        return False

if __name__ == "__main__":
    main() 