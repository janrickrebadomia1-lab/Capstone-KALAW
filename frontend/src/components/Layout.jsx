import React, { useState, useRef, useEffect } from "react";
import Sidebar from "./Sidebar";
import MainContent from "./MainContent";
import Navbar from "./Navbar";
import "../styles/Layout.css";

const Layout = () => {
  // Sidebar open/close state lives HERE — not in App.jsx
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const [activeMenu, setActiveMenu] = useState("Chatbot");
  const [input, setInput] = useState("");

  const [chatHistory, setChatHistory] = useState(() => {
    try {
      const saved = localStorage.getItem("chatHistory");
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  const [activeChatId, setActiveChatId] = useState(null);
  const chatRef = useRef(null);

  /*
  useEffect(() => {
    localStorage.setItem("chatHistory", JSON.stringify(chatHistory));
  }, [chatHistory]);
  */

  // Auto-close sidebar when screen grows past tablet breakpoint
  useEffect(() => {
    const mq = window.matchMedia("(min-width: 769px)");
    const handler = (e) => { if (e.matches) setSidebarOpen(false); };
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);

  const handleDeleteChat = (chatId) => {
    setChatHistory((prev) => prev.filter((chat) => chat.id !== chatId));
  };

  const handleSendQuestion = (question) => {
    if (chatRef.current) chatRef.current.sendQuestion(question);
  };

  const handleNewChat = () => {
    setActiveChatId(null);
    setInput("");
    setSidebarOpen(false);
    if (chatRef.current) {
      chatRef.current.newChat();
      chatRef.current.scrollToTop?.();
      chatRef.current.focusInput?.();
    }
  };

  const handleLoadChat = (chatId) => {
    setActiveChatId(chatId);
    setSidebarOpen(false);
    const found = chatHistory.find((c) => c.id === chatId);
    if (!found) return;
    if (chatRef.current) {
      chatRef.current.loadChat(chatId, chatHistory);
    }
  };

  return (
    <div className="layout-container">
      <Navbar onHamburgerClick={() => setSidebarOpen((prev) => !prev)} />

      <div className="app-container">
        {/* Dark overlay — tapping closes the sidebar on mobile */}
        {sidebarOpen && (
          <div
            className="sidebar-overlay open"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        <Sidebar
          isOpen={sidebarOpen}
          handleSendQuestion={handleSendQuestion}
          chatHistory={chatHistory}
          activeChatId={activeChatId}
          onNewChat={handleNewChat}
          onLoadChat={handleLoadChat}
          onDeleteChat={handleDeleteChat}
        />

        <MainContent
          activeMenu={activeMenu}
          input={input}
          setInput={setInput}
          chatRef={chatRef}
          setChatHistory={setChatHistory}
          setActiveChatId={setActiveChatId}
        />
      </div>
    </div>
  );
};

export default Layout;