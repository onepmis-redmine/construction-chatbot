import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
from typing import Dict, List, Tuple
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/faq_enhancement.log'),
        logging.StreamHandler()
    ]
)

class FAQEnhancer:
    def __init__(self):
        load_dotenv()
        # API 키 확인
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        logging.info("API Key loaded successfully")
        
        # Gemini API 설정
        genai.configure(api_key=api_key)
        
        # 사용 가능한 모델 목록 확인
        try:
            models = genai.list_models()
            logging.info(f"Available models: {[model.name for model in models]}")
        except Exception as e:
            logging.error(f"Error listing models: {e}")
        
        # 모델 설정
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash-latest')
            logging.info("Gemini model initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing model: {e}")
            raise

    def read_faq_excel(self, file_path: str) -> pd.DataFrame:
        """FAQ 엑셀 파일을 읽어옵니다."""
        try:
            df = pd.read_excel(file_path)
            logging.info(f"Successfully read FAQ file: {file_path}")
            return df
        except Exception as e:
            logging.error(f"Error reading FAQ file: {e}")
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
            "구체적인 예시 1",
            "구체적인 예시 2"
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
"""
        
        try:
            response = self.model.generate_content(prompt)
            # JSON 응답 파싱
            # Gemini의 응답에서 JSON 부분만 추출
            json_str = response.text
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].strip()
            
            enhanced_content = json.loads(json_str)
            logging.info(f"Successfully enhanced Q&A pair: {question}")
            return enhanced_content
            
        except Exception as e:
            logging.error(f"Error enhancing Q&A pair: {e}")
            return None

    def process_faq_file(self, input_file: str, output_file: str):
        """전체 FAQ 파일을 처리합니다."""
        df = self.read_faq_excel(input_file)
        enhanced_data = []

        for idx, row in df.iterrows():
            question = row['질문']
            answer = row['답변']
            
            enhanced = self.enhance_qa_pair(question, answer)
            if enhanced:
                enhanced_data.append(enhanced)
                
            # 진행 상황 출력
            logging.info(f"Processed {idx + 1}/{len(df)} QA pairs")

        # 결과를 새로운 엑셀 파일로 저장
        output_df = pd.DataFrame(enhanced_data)
        output_df.to_excel(output_file, index=False)
        logging.info(f"Successfully saved enhanced FAQ to: {output_file}")

def main():
    enhancer = FAQEnhancer()
    
    # FAQ 파일 경로 수정
    input_file = "../docs/qa_pairs.xlsx"
    output_file = "../docs/enhanced_qa_pairs.xlsx"
    
    # FAQ 개선 실행
    enhancer.process_faq_file(input_file, output_file)

if __name__ == "__main__":
    main() 