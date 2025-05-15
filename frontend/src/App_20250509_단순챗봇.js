import React, { useState } from "react";
import axios from "axios";
import "./App.css";

<div className="bg-blue-500 text-white p-4">테스트</div>

function App() {
  const [question, setQuestion] = useState("비밀번호를 잃어버렸어. 어떻게 해?");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setAnswer("");
    setSources([]);
    setLoading(true); // 로딩 시작
    console.log("Loading set to true");
    
    try {
      const response = await axios.post("http://localhost:8000/ask", {
        query: question,
      });

      // 최소 1초 로딩 보장
      // const startTime = Date.now();
      // const elapsedTime = Date.now() - startTime;
      // const minLoadingTime = 1000; // 1초
      // if (elapsedTime < minLoadingTime) {
      //   await new Promise((resolve) => setTimeout(resolve, minLoadingTime - elapsedTime));
      // }

      // 인위적 지연 추가 (2초)
      // await new Promise((resolve) => setTimeout(resolve, 2000));

      setAnswer(response.data.answer);
      setSources(Array.isArray(response.data.sources) ? response.data.sources : []);

    } catch (error) {
      setAnswer("❌ 서버 응답 중 오류가 발생했습니다.");
      console.error(error);
    }

    setLoading(false); // 로딩 끝
    console.log("Loading set to false");
  };

  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "flex-start",
      minHeight: "100vh",
      padding: "40px",
      backgroundColor: "#f9f9f9",
      fontFamily: "Arial, sans-serif"
    }}>

      <h1 style={{ fontSize: "32px", marginBottom: "20px" }}>🏗️ 건설 매뉴얼 챗봇</h1>
      <form onSubmit={handleSubmit} style={{ display: "flex", gap: "10px" }}>
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            style={{
              width: "400px",
              padding: "10px",
              fontSize: "16px",
              borderRadius: "5px",
              border: "1px solid #ccc"
            }}
          />
          <button
            type="submit"
            style={{
              padding: "10px 20px",
              fontSize: "16px",
              backgroundColor: "#007bff",
              color: "white",
              border: "none",
              borderRadius: "5px",
              cursor: "pointer"
            }}
          >
            질문하기
          </button>
      </form>

      <div style={{ marginTop: "30px", width: "600px", textAlign: "left" }}>
        {loading ? (
          <div className="spinner" style={{ margin: "40px auto" }} />
        ) : (
          <>
            <strong style={{ fontSize: "18px" }}>답변:</strong>
            <p>{answer}</p>

            {sources.length > 0 && (
              <div>
                <h4>출처:</h4>
                <ul>
                  {sources.map((src, i) => (
                    <li key={i}>{src}</li>
                  ))}
                </ul>
              </div>
            )}
          </>
        )}
      </div>              
    </div>
  );
}

export default App;