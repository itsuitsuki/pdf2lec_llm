import React, { useState, useEffect, useRef } from "react";
import { axiosInstance } from "../api/axios";
import { useParams } from "react-router-dom";

const Chatbot: React.FC = () => {
  const { pdfId } = useParams<{ pdfId: string }>();
  const [messages, setMessages] = useState<{ sender: string; text: string }[]>([
    {
      sender: "bot",
      text: "Hi! I'm your lecture assistant. Feel free to ask any questions about the lecture or general topics!",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (input.trim() === "") return;

    setMessages((prev) => [...prev, { sender: "user", text: input }]);
    const userMessage = input;
    setInput("");
    setIsLoading(true);

    try {
      const response = await axiosInstance.post("/chatbot", {
        message: userMessage,
        pdfId: pdfId,
      });

      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: response.data.reply },
      ]);
    } catch (error) {
      console.error("Error:", error);
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "Sorry, there was a problem with my response" },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div
      style={{
        width: "80%",
        height: "100%",
        backgroundColor: "white",
        boxShadow: "0 0 10px rgba(0,0,0,0.1)",
        borderRadius: "10px",
        zIndex: 1000,
        padding: "20px",
        display: "flex",
        flexDirection: "column",
        marginLeft: "2%",
      }}
    >
      <h2 style={{ marginBottom: "15px", fontSize: "27px" }}>
        Lecture Assistant
      </h2>
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          border: "1px solid #ccc",
          marginBottom: "20px",
          borderRadius: "10px",
          padding: "7px",
        }}
      >
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`message ${msg.sender === "bot" ? "bot" : "user"}`}
            style={{
              marginBottom: "0.5rem",
              // marginBottom: msg.sender === "bot" ? "1rem" : "0", 
            }}
          >
            <div className="message-content">
              <strong
                style={{
                  color: msg.sender === "bot" ? "darkblue" : "inherit",
                }}
              >
                {msg.sender === "bot" ? "Assistant" : "You"}:
              </strong>{" "}
              {msg.text}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="message bot">
            <div className="message-content">
              <em>Typing...</em>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <div
        style={{
          marginTop: "auto",
          display: "flex",
          alignItems: "center",
        }}
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask about the lecture..."
          disabled={isLoading}
          style={{
            width: "100%",
            padding: "10px",
            border: "1px solid #ccc",
            borderRadius: "5px",
            marginRight: "10px",
          }}
        />
        <button
          onClick={sendMessage}
          disabled={isLoading || input.trim() === ""}
          style={{
            padding: "10px 20px",
            backgroundColor: "#007bff",
            color: "#fff",
            border: "none",
            borderRadius: "5px",
            cursor: "pointer",
          }}
        >
          Send
        </button>
      </div>
    </div>
  );
};

export default Chatbot;
