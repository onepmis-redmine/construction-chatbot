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
genai.configure(api_key=GOOGLE_API_KEY)

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
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
        logger.info("QAProcessor initialized successfully")
        logger.info(f"EXCEL_PATH 확인: {EXCEL_PATH}, 존재여부: {EXCEL_PATH.exists()}")

    def read_qa_excel(self) -> pd.DataFrame:
        """FAQ 엑셀 파일을 읽어옵니다."""
        try:
            if not EXCEL_PATH.exists():
                logger.error(f"FAQ 엑셀 파일이 없습니다: {EXCEL_PATH}")
                raise FileNotFoundError(f"FAQ 엑셀 파일이 없습니다: {EXCEL_PATH}")
                
            df = pd.read_excel(EXCEL_PATH)
            logger.info(f"Successfully read FAQ file: {EXCEL_PATH}")
            return df
        except Exception as e:
            logger.error(f"Error reading FAQ file: {e}")
            raise

    def enhance_qa_pair(self, question: str, answer: str) -> Dict:
        """Gemini를 사용하여 질문-답변 쌍을 구조화합니다."""
        prompt = f"""
다음 질문-답변 쌍을 구조화된 형식으로 변환해주세요.
결과는 반드시 올바른 JSON 형식이어야 합니다.

원본 질문: {question}
원본 답변: {answer}

다음 구조로 JSON을 생성해주세요:
{{
    "question_variations": [
        "변형된 질문 1",
        "변형된 질문 2",
        "변형된 질문 3"
    ],
    "structured_answer": {{
        "basic_rules": [
            "기본 규칙/조건 1",
            "기본 규칙/조건 2"
        ],
        "examples": [
            {{
                "scenario": "예시 상황 설명",
                "result_a": "A의 결과",
                "result_b": "B의 결과"
            }},
            {{
                "scenario": "다른 예시 상황",
                "result_c": "C의 결과"
            }}
        ],
        "cautions": [
            "주의사항 1",
            "주의사항 2"
        ]
    }},
    "keywords": [
        "키워드1",
        "키워드2",
        "키워드3"
    ]
}}

응답은 반드시 위와 같은 형식의 유효한 JSON이어야 합니다.
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
            logger.error(f"Error enhancing Q&A pair: {e}")
            return None

    def process_and_enhance_qa(self):
        """FAQ를 읽고 구조화합니다."""
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
                        'question_variations': json.dumps(enhanced['question_variations'], ensure_ascii=False),
                        'structured_answer': json.dumps(enhanced['structured_answer'], ensure_ascii=False),
                        'keywords': json.dumps(enhanced['keywords'], ensure_ascii=False)
                    }
                    enhanced_data.append(enhanced)
                    logger.info(f"Successfully processed Q&A pair {idx + 1}: {question}")
                else:
                    logger.error(f"Missing required fields in enhanced data for question: {question}")
                
            logger.info(f"Processed {idx + 1}/{len(df)} QA pairs")

        if not enhanced_data:
            logger.error("No FAQ data was successfully processed")
            return []

        # 결과를 새로운 엑셀 파일로 저장
        output_df = pd.DataFrame(enhanced_data)
        
        # 저장 전 데이터 확인
        logger.debug(f"Enhanced data before saving:\n{output_df.to_string()}")
        
        try:
            output_df.to_excel(ENHANCED_FAQ_PATH, index=False)
            logger.info(f"Successfully saved enhanced FAQ to: {ENHANCED_FAQ_PATH}")
        except Exception as e:
            logger.error(f"Error saving enhanced FAQ: {e}")
            
        return enhanced_data

def process_qa_file():
    """FAQ 파일을 처리하고 구조화합니다."""
    processor = QAProcessor()
    enhanced_data = processor.process_and_enhance_qa()
    logger.info(f"FAQ 구조화 완료. 총 {len(enhanced_data)}개의 FAQ가 처리되었습니다.")
    return enhanced_data

def main():
    """메인 함수"""
    try:
        process_qa_file()
        logger.info("FAQ 구조화 처리가 완료되었습니다.")
        return True
    except Exception as e:
        logger.error(f"FAQ 구조화 처리 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    main() 