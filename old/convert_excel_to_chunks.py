import pandas as pd
import json
from pathlib import Path

# 엑셀 파일 경로
excel_path = Path("docs/qa_pairs.xlsx")
output_path = Path("chunks/from_excel.json")

# 출력 폴더 만들기
output_path.parent.mkdir(parents=True, exist_ok=True)

# 엑셀 로드
df = pd.read_excel(excel_path)

chunks = []
for idx, row in df.iterrows():
    question = str(row["질문"]).strip()
    answer = str(row["답변"]).strip()
    source = str(row.get("출처", "엑셀_QA"))

    if question and answer:
        chunks.append({
            "text": f"Q: {question}\nA: {answer}",
            "source": source
        })

# JSON으로 저장
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"✅ Q&A {len(chunks)}개 변환 완료 → {output_path}")
