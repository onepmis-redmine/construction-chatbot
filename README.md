#파일 다시 chunks
cd ..
python .\embedding_chunks.py

#FastAPI 서버 실행하기
cd api_server
uvicorn main:app --reload
#접속 url
http://127.0.0.1:8000/docs



# 다음 해야할꺼
📌 이제 다음 단계는?
1. 💬 답변에 출처 보여주기 (선택)  << 완료>>
    어떤 문서에서 가져온 건지 문단 ID나 파일명 표시   # 출처 보여주긴 하는데 영 이상하다.
2. 🧠 문서 자동 업데이트 (선택)
    raw_texts/, chunks/ 자동 갱신 스크립트 만들기
3. 🌐 웹 UI 붙이기 (React, Vue, 또는 Streamlit)
- React 프로젝트 만들기
- VS Code에서 실행하기
- FastAPI와 연결하기

# React 프론트 프로젝트 실행
cd .\frontend\
npm start

http://localhost:3000
비밀번호를 잃어버렸어. 어떻게 해?


4. 🛡️ 오류 방지 보완
    OpenRouter 응답 없을 때 대비 try-except
    요청 속도 제한 고려


    🔍 출처를 클릭 가능하게 만들기

📄 업로드한 문서 목록 보기

🧪 UI 개선 (Material UI, Tailwind 등)

💬 챗봇 형식 변경 (대화 방식)

🔁 GPT 응답 캐싱 등
