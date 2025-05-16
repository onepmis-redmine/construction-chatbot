import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import "./App.css";

// API 기본 URL 설정 - 개발/프로덕션 환경에 따라 다른 URL 사용
const API_BASE_URL = process.env.NODE_ENV === 'development' 
  ? "http://localhost:8000" 
  : "https://construction-chatbot-api.onrender.com";

console.log(`운영 모드: ${process.env.NODE_ENV}, API URL: ${API_BASE_URL}`);

function App() {
  const [messages, setMessages] = useState(() => {
    // 대화 기록 로드 (localStorage에서 복원)
    const saved = localStorage.getItem("chatHistory");
    return saved
      ? JSON.parse(saved)
      : [{ role: "bot", content: "안녕하세요! 건설정보시스템에 대해 궁금한 점을 말씀해주세요.", sources: [] }];
  });
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [isDevMode, setIsDevMode] = useState(process.env.NODE_ENV === 'development');
  const [showDebugInfo, setShowDebugInfo] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null); // 입력 필드 참조

  // 버튼 텍스트 고정
  const sendButtonText = "질문하기";
  const clearButtonText = "대화 초기화";

  // 메시지 변경 시 localStorage에 저장
  useEffect(() => {
    localStorage.setItem("chatHistory", JSON.stringify(messages));
  }, [messages]);

  // 메시지 추가 시 자동 스크롤
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // 로딩 상태 변경 시 포커스 설정
  useEffect(() => {
    if (!loading && inputRef.current) {
      console.log("포커스 설정 시도");
      inputRef.current.focus();
    }
  }, [loading]);

  const toggleDebugInfo = () => {
    setShowDebugInfo(!showDebugInfo);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    // 디버깅: 전송된 질문 로그
    console.log("전송된 질문:", input);

    // 사용자 메시지 추가
    const userMessage = { role: "user", content: input, sources: [] };
    setMessages((prev) => {
      const newMessages = [...prev, userMessage];
      console.log("대화 기록에 추가된 메시지:", newMessages[newMessages.length - 1]);
      return newMessages;
    });
    setInput(""); // 입력 필드 비우기
    setLoading(true);

    try {
      // 개발/프로덕션 환경에 따라 다른 API URL 사용
      const response = await axios.post(`${API_BASE_URL}/ask`, {
        query: input,
      });

      // 디버그 정보 로깅
      if (response.data.is_dev) {
        console.log("디버그 정보:", response.data.debug_info);
      }

      // 챗봇 답변 추가
      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          content: response.data.answer,
          sources: Array.isArray(response.data.sources) ? response.data.sources : [],
          debug_info: response.data.debug_info || {},
          is_dev: response.data.is_dev || false
        },
      ]);
    } catch (error) {
      // 개선된 에러 메시지
      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          content: "죄송해요, 서버에 문제가 생겼어요. 잠시 후 다시 시도해주세요!",
          sources: [],
          error: error.toString(),
          is_dev: isDevMode
        },
      ]);
      console.error("서버 에러:", error);
    }

    setLoading(false);
  };

  // 대화 초기화
  const clearChat = () => {
    setMessages([
      { role: "bot", content: "안녕하세요! 건설정보시스템에 대해 궁금한 점을 말씀해주세요.", sources: [] },
    ]);
    setInput(""); // 초기화 시 입력 필드 비우기
    setLoading(false); // 명시적으로 로딩 상태 해제
    console.log("대화 초기화 버튼 클릭");
  };

  return (
    <div className="app-container">
      <h1 className="app-title">🏗️ 건설 매뉴얼 챗봇</h1>
      
      {isDevMode && (
        <div className="dev-controls">
          <button onClick={toggleDebugInfo} className="debug-button">
            {showDebugInfo ? "디버그 정보 숨기기" : "디버그 정보 보기"}
          </button>
          <span className="dev-badge">개발 모드</span>
        </div>
      )}
      
      <div className="chat-container">
        <div className="messages">
          {messages.map((msg, index) => (
            <div
              key={index}
              className={`message ${msg.role === "user" ? "user-message" : "bot-message"}`}
            >
              <div className="message-content">
                {msg.content}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="sources">
                    <strong>출처:</strong>
                    <ul>
                      {msg.sources.map((src, i) => (
                        <li key={i}>{src}</li>
                      ))}
                    </ul>
                  </div>
                )}
                
                {showDebugInfo && msg.is_dev && msg.debug_info && (
                  <div className="debug-info">
                    <details>
                      <summary>디버그 정보</summary>
                      <pre>{JSON.stringify(msg.debug_info, null, 2)}</pre>
                    </details>
                  </div>
                )}
                
                {showDebugInfo && msg.is_dev && msg.error && (
                  <div className="debug-info">
                    <details>
                      <summary>오류 정보</summary>
                      <pre>{msg.error}</pre>
                    </details>
                  </div>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="message bot-message">
              <div className="spinner" />
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        <div className="input-container">
          <form onSubmit={handleSubmit} className="input-form">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="질문을 입력하세요..."
              disabled={loading}
              className="input-field"
              ref={inputRef}
            />
            <button type="submit" disabled={loading} className="send-button">
              {sendButtonText}
            </button>
          </form>
          <button onClick={clearChat} className="clear-button">
            {clearButtonText}
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;