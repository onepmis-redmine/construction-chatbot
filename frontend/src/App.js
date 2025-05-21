import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import "./App.css";
import dayjs from "dayjs";

// API 기본 URL 설정 - 개발/프로덕션 환경에 따라 다른 URL 사용
const API_BASE_URL = process.env.NODE_ENV === 'development' 
  ? "http://localhost:8000" 
  : "https://construction-chatbot-api.onrender.com";

console.log(`운영 모드: ${process.env.NODE_ENV}, API URL: ${API_BASE_URL}`);

// 구조화된 답변을 포맷팅하는 함수
const formatStructuredAnswer = (structuredAnswer) => {
  if (!structuredAnswer) return "답변 데이터가 없습니다.";
  
  let formatted = "답변:\n\n";
  
  // 기본 규칙 포맷팅
  if (structuredAnswer.basic_rules && structuredAnswer.basic_rules.length > 0) {
    formatted += "• 기본 규칙:\n";
    structuredAnswer.basic_rules.forEach(rule => {
      formatted += `  - ${rule}\n`;
    });
    formatted += "\n";
  }
  
  // 예시 포맷팅
  if (structuredAnswer.examples && structuredAnswer.examples.length > 0) {
    formatted += "• 예시:\n";
    structuredAnswer.examples.forEach(example => {
      if (typeof example === 'object') {
        // 딕셔너리 형태의 예시는 시나리오와 결과로 구분하여 처리
        Object.entries(example).forEach(([key, value]) => {
          if (key.toLowerCase().startsWith('scenario')) {
            formatted += `  📌 ${value}\n`;
          } else if (key.toLowerCase().startsWith('result')) {
            formatted += `      ➡️ ${value}\n`;
          } else {
            formatted += `      • ${key}: ${value}\n`;
          }
        });
      } else {
        formatted += `  - ${example}\n`;
      }
    });
    formatted += "\n";
  }
  
  // 주의사항 포맷팅
  if (structuredAnswer.cautions && structuredAnswer.cautions.length > 0) {
    formatted += "• 주의사항:\n";
    structuredAnswer.cautions.forEach(caution => {
      formatted += `  - ${caution}\n`;
    });
    formatted += "\n";
  }
  
  return formatted;
};

function App() {
  const [messages, setMessages] = useState(() => {
    // 대화 기록 로드 (localStorage에서 복원)
    const saved = localStorage.getItem("chatHistory");
    return saved
      ? JSON.parse(saved)
      : [{ role: "bot", content: "안녕하세요! 건설정보시스템에 대해 궁금하신 점을 말씀해주세요.", sources: [] }];
  });
  const [sessionId, setSessionId] = useState(() => {
    // 세션 ID 로드 (localStorage에서 복원)
    return localStorage.getItem("sessionId") || null;
  });
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const isDevMode = process.env.NODE_ENV === 'development';  // 상태 변수에서 상수로 변경
  const [showDebugInfo, setShowDebugInfo] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null); // 입력 필드 참조
  const fileInputRef = useRef(null); // 파일 입력 필드 참조
  
  // 파일 업로드 관련 상태
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [processingStatus, setProcessingStatus] = useState('');
  const [showUploadPanel, setShowUploadPanel] = useState(false);

  // 버튼 텍스트 고정
  const sendButtonText = "질문하기";
  const clearButtonText = "대화 초기화";
  const downloadButtonText = "질문 다운로드";

  // 메시지와 세션 ID 변경 시 localStorage에 저장
  useEffect(() => {
    localStorage.setItem("chatHistory", JSON.stringify(messages));
    if (sessionId) {
      localStorage.setItem("sessionId", sessionId);
    }
  }, [messages, sessionId]);

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

  // 세션 ID가 변경될 때 대화 기록 로드
  useEffect(() => {
    if (sessionId) {
      loadSessionMessages(sessionId);
    }
  }, [sessionId]);

  const loadSessionMessages = async (sid) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/session/${sid}`);
      if (response.data.messages && response.data.messages.length > 0) {
        setMessages(response.data.messages.map(msg => ({
          role: msg.role === "assistant" ? "bot" : "user",
          content: msg.content,
          sources: []
        })));
      }
    } catch (error) {
      console.error("세션 메시지 로드 실패:", error);
    }
  };

  const toggleDebugInfo = () => {
    setShowDebugInfo(!showDebugInfo);
  };

  const toggleUploadPanel = () => {
    setShowUploadPanel(!showUploadPanel);
  };

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setUploadStatus('');
    setProcessingStatus('');
  };

  const uploadFile = async () => {
    if (!selectedFile) {
      setUploadStatus('파일을 선택해주세요.');
      return;
    }

    // 파일 확장자 확인
    const fileExt = selectedFile.name.split('.').pop().toLowerCase();
    if (fileExt !== 'xlsx' && fileExt !== 'xls') {
      setUploadStatus('Excel 파일(.xlsx 또는 .xls)만 업로드 가능합니다.');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);
    
    setUploadStatus('업로드 중...');
    
    try {
      const response = await axios.post(`${API_BASE_URL}/upload-excel`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      setUploadStatus(`업로드 완료: ${response.data.message}`);
      // 파일 선택 초기화
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      setSelectedFile(null);
      
      // 업로드가 완료되면 처리 버튼을 강조
      const processButton = document.getElementById('process-excel-button');
      if (processButton) {
        processButton.classList.add('highlight-button');
        setTimeout(() => {
          processButton.classList.remove('highlight-button');
        }, 3000);
      }
    } catch (error) {
      console.error('파일 업로드 오류:', error);
      setUploadStatus(`업로드 실패: ${error.response?.data?.detail || error.message}`);
    }
  };

  const processExcel = async () => {
    setProcessingStatus('FAQ 구조화 처리 및 임베딩 중...');
    
    try {
      const response = await axios.post(`${API_BASE_URL}/process-excel`);
      
      if (response.data.success) {
        setProcessingStatus(`FAQ 처리 완료: ${response.data.message}`);
        
        // 성공 메시지를 챗봇 메시지로 추가
        setMessages(prev => [
          ...prev,
          {
            role: "bot",
            content: `FAQ 데이터가 성공적으로 처리되었습니다. 이제 질문을 입력해보세요.`,
            sources: []
          }
        ]);
        
        // 업로드 패널 닫기
        setShowUploadPanel(false);
      } else {
        setProcessingStatus(`FAQ 처리 실패: ${response.data.message}`);
      }
    } catch (error) {
      console.error('FAQ 처리 오류:', error);
      setProcessingStatus(`FAQ 처리 실패: ${error.response?.data?.message || error.message}`);
    }
  };

  // 질문 다운로드 함수
  const downloadQuestions = () => {
    // 다운로드 URL 생성
    const downloadUrl = `${API_BASE_URL}/download-questions`;
    
    // 새 탭에서 다운로드 링크 열기
    window.open(downloadUrl, '_blank');
  };

  // enhanced_qa_pairs.xlsx 다운로드 함수
  const downloadEnhancedQA = () => {
    // 현재 시간 포맷 (예: 202505211728)
    const now = dayjs().format("YYYYMMDDHHmm");
    const filename = `enhanced_qa_pairs_${now}.xlsx`;
    const url = `${API_BASE_URL}/download-enhanced-qa?filename=${filename}`;
    // a 태그를 동적으로 생성하여 다운로드 트리거
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    console.log("전송된 질문:", input);

    // 사용자 메시지 추가
    const userMessage = { role: "user", content: input, sources: [] };
    setMessages((prev) => {
      const newMessages = [...prev, userMessage];
      console.log("대화 기록에 추가된 메시지:", newMessages[newMessages.length - 1]);
      return newMessages;
    });
    setInput("");
    setLoading(true);

    try {
      // session_id가 null이나 undefined인 경우 payload에서 제외
      const payload = { query: input };
      if (sessionId) {
        payload.session_id = sessionId;
      }

      console.log("서버에 전송하는 payload:", JSON.stringify(payload));

      const response = await axios.post(`${API_BASE_URL}/ask`, payload);

      // 세션 ID가 새로 생성된 경우 저장
      if (response.data.session_id && response.data.session_id !== sessionId) {
        setSessionId(response.data.session_id);
      }

      if (response.data.is_dev) {
        console.log("디버그 정보:", response.data.debug_info);
      }
      
      // 응답 내용 처리
      let content;
      if (response.data.structured_answer) {
        // 구조화된 답변이 있으면 포맷팅
        content = formatStructuredAnswer(response.data.structured_answer);
      } else if (response.data.answer) {
        // 일반 텍스트 답변
        content = response.data.answer;
      } else {
        // 답변 없음
        content = "죄송합니다. 답변을 찾을 수 없습니다.";
      }

      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          content: content,
          sources: Array.isArray(response.data.sources) ? response.data.sources : [],
          debug_info: response.data.debug_info || {},
          is_dev: response.data.is_dev || false,
          type: response.data.type || "unknown"
        },
      ]);
    } catch (error) {
      console.error("서버 에러:", error);
      console.error("에러 상세 정보:", error.response?.data || error.message);
      
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
    }

    setLoading(false);
  };

  // 대화 초기화
  const clearChat = () => {
    setMessages([
      { role: "bot", content: "안녕하세요! 건설정보시스템에 대해 궁금하신 점을 말씀해주세요.", sources: [] },
    ]);
    setSessionId(null);
    localStorage.removeItem("sessionId");
    setInput("");
    setLoading(false);
    console.log("대화 초기화 버튼 클릭");
  };

  return (
    <div className="app-container">
      <h1 className="app-title">🏗️ 건설 매뉴얼 챗봇</h1>
      
      <div className="controls-bar">
        {isDevMode && (
          <div className="dev-controls">
            <button onClick={toggleDebugInfo} className="debug-button">
              {showDebugInfo ? "디버그 정보 숨기기" : "디버그 정보 보기"}
            </button>
            <span className="dev-badge">개발 모드</span>
          </div>
        )}
        
        <button 
          onClick={toggleUploadPanel} 
          className="upload-button"
          title="Excel 파일을 업로드하여 FAQ 데이터베이스를 업데이트합니다"
        >
          {showUploadPanel ? "업로드 패널 숨기기" : "FAQ 업로드"}
        </button>
        
        <button 
          onClick={downloadQuestions} 
          className="download-button"
          title="사용자 질문 데이터를 Excel 파일로 다운로드합니다"
        >
          {downloadButtonText}
        </button>
      </div>
      
      {showUploadPanel && (
        <div className="upload-panel">
          <h3>FAQ 파일 업로드</h3>
          
          <div className="workflow-steps">
            <div className="workflow-step">
              <div className="step-number">1</div>
              <div className="step-text">엑셀 파일 업로드</div>
            </div>
            <div className="workflow-step">
              <div className="step-number">2</div>
              <div className="step-text">FAQ 구조화 처리</div>
            </div>
            <div className="workflow-step">
              <div className="step-number">3</div>
              <div className="step-text">임베딩 생성</div>
            </div>
          </div>
          
          <div className="upload-instructions">
            <p><strong>FAQ 관리 방법:</strong></p>
            <ol>
              <li>질문-답변이 포함된 엑셀 파일(.xlsx)을 업로드합니다.</li>
              <li>'FAQ 구조화 처리 시작' 버튼을 클릭하여 자연어 처리와 벡터 임베딩을 생성합니다.</li>
              <li>처리가 완료되면 챗봇이 FAQ를 활용할 수 있습니다.</li>
            </ol>
          </div>
          
          <div className="file-upload-container">
            <input 
              type="file" 
              className="file-input" 
              onChange={handleFileChange} 
              ref={fileInputRef}
              accept=".xlsx,.xls"
            />
            <button 
              className="upload-action-button" 
              onClick={uploadFile}
              disabled={!selectedFile || uploadStatus.includes('업로드 중')}
            >
              {uploadStatus.includes('업로드 중') ? '업로드 중...' : '파일 업로드'}
            </button>
          </div>
          
          {uploadStatus && <div className="status-message">{uploadStatus}</div>}
          
          <div className="process-container">
            <button 
              onClick={processExcel} 
              className="process-button"
              disabled={processingStatus.includes('FAQ 처리 중')}
              id="process-excel-button"
            >
              FAQ 구조화 처리 시작
            </button>
            {processingStatus && <div className="status-message">{processingStatus}</div>}
          </div>
          
          <button
            onClick={downloadEnhancedQA}
            className="download-excel-button"
            title="구조화된 FAQ 엑셀 파일을 다운로드합니다"
          >
            구조화 FAQ 다운로드
          </button>
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