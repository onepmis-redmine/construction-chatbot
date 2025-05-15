import React, { useState } from "react";
import axios from "axios";

function App() {
  const [question, setQuestion] = useState("비밀번호를 잃어버렸어. 어떻게 해?");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState("");
  

  const handleSubmit = async (e) => {
    e.preventDefault();
    setAnswer("잠시만 기다려 주세요...");

    try {
      const response = await axios.post("http://localhost:8000/ask", {
        query: question,
      });

      setAnswer(response.data.answer);
      setSources(Array.isArray(response.data.sources) ? response.data.sources : []);

    } catch (error) {
      setAnswer("❌ 서버 응답 중 오류가 발생했습니다.");
      console.error(error);
    }
  };

  return (
    <div style={{ padding: "20px", fontFamily: "Arial" }}>
      <h1>건설 매뉴얼 챗봇</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={question} //비밀번호를 잃어버렸어. 어떻게 해?
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="질문을 입력하세요"
          style={{ width: "300px", marginRight: "10px" }}
        />
        <button type="submit">질문하기</button>
      </form>
      <div style={{ marginTop: "20px" }}>
        <strong>답변:</strong>
        <p>{answer}</p>

        {/* ✅ 출처 표시 영역 */}
        {sources.length > 0 && (
          <div>
            <h4>출처:</h4>
            <ul>
              {Array.isArray(sources) && sources.map((src, i) => (
                <li key={i}>{src}</li>
              ))}
            </ul>
          </div>
        )}



      </div>
    </div>
  );
}

export default App;
