import React, { useState, useEffect, useRef } from "react";
import "../styles/Layout.css";

const COMMON_QUESTIONS = [
  "What are the criteria for faculty promotion?",
  "What is the regular teaching load for permanent faculty?",
  "What are the procedures for filing a grievance?",
];

const Sidebar = ({
  isOpen,
  handleSendQuestion,
  chatHistory = [],
  activeChatId = null,
  onNewChat,
  onLoadChat,
  onDeleteChat,
  onRenameChat,
}) => {
  const [menuOpenId, setMenuOpenId] = useState(null);
  const [alertState, setAlertState] = useState({ visible: false, chatId: null, chatTitle: "" });
  const menuRef = useRef(null);

  // Close three-dot menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setMenuOpenId(null);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleDeleteClick = (chat) => {
    setAlertState({ visible: true, chatId: chat.id, chatTitle: chat.title });
    setMenuOpenId(null);
  };

  const confirmDelete = () => {
    onDeleteChat(alertState.chatId);
    setAlertState({ visible: false, chatId: null, chatTitle: "" });
  };

  const cancelDelete = () => {
    setAlertState({ visible: false, chatId: null, chatTitle: "" });
  };

  return (
    <>
      <div className={`sidebar ${isOpen ? "open" : ""}`}>
        <button className="new-chat-btn" onClick={onNewChat}>+ New Chat</button>

        {/* Common Questions */}
        <div className="sidebar-container">
          <div className="sidebar-examples">
            <h4>Common Questions</h4>
            {COMMON_QUESTIONS.map((q, i) => (
              <p key={i} onClick={() => handleSendQuestion(q)} title={q}>
                {q}
              </p>
            ))}
          </div>
        </div>

        {/* Recent Chats */}
        <div className="recent-container">
          <div className="recent-chats">
            <h4>Recent Chats</h4>
            {chatHistory.length === 0 ? (
              <p className="sidebar-empty">No recent chats yet.</p>
            ) : (
              <div ref={menuRef}>
                {chatHistory.map((chat) => (
                  <div
                    key={chat.id}
                    className={`sidebar-chat-item-wrapper ${activeChatId === chat.id ? "active" : ""}`}
                  >
                    <p
                      className={`sidebar-chat-item ${activeChatId === chat.id ? "active" : ""}`}
                      onClick={() => onLoadChat(chat.id)}
                      title={chat.title}
                    >
                      {chat.title}
                    </p>

                    {/* Three-dot menu */}
                    <div className="chat-menu-container">
                      <button
                        className="chat-menu-dot"
                        onClick={() =>
                          setMenuOpenId(menuOpenId === chat.id ? null : chat.id)
                        }
                        title="Actions"
                      >
                        •••
                      </button>

                      <div className={`chat-menu-mini ${menuOpenId === chat.id ? "open" : ""}`}>
                        <button onClick={() => handleDeleteClick(chat)}>Delete</button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="sidebar-footer">
          <small>Ask about CPSU faculty policies, workloads, and career advancement.</small>
        </div>
      </div>

      {/* Delete confirmation alert — rendered outside sidebar so it overlays everything */}
      {alertState.visible && (
        <div className="alert-overlay">
          <div className="alert-box">
            <p>
              Are you sure you want to delete{" "}
              <strong>"{alertState.chatTitle}"</strong>?
            </p>
            <div className="alert-buttons">
              <button className="alert-btn cancel" onClick={cancelDelete}>
                Cancel
              </button>
              <button className="alert-btn confirm" onClick={confirmDelete}>
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default Sidebar;