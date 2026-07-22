import React from "react";
import Chat from "./Chat";

const MainContent = ({ activeMenu, input, setInput, chatRef, setChatHistory, setActiveChatId }) => {
  return (
    <div className="main-content">
      {activeMenu === "Chatbot" && (
        <Chat
          ref={chatRef}
          sidebarInput={input}
          setSidebarInput={setInput}
          setChatHistory={setChatHistory}
          setActiveChatId={setActiveChatId} // pass this to Chat.jsx
        />
      )}
      {activeMenu === "Announcement" && <div>Announcements will show here</div>}
    </div>
  );
};

export default MainContent;