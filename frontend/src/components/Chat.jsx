import React, { useState, useRef, useEffect, forwardRef, useImperativeHandle } from "react";
import Kalaw from "../assets/kalaw.jpg";
import "../styles/Chat.css";

const SUGGESTIONS = [
  { icon: "🎓", title: "Promotion Guidelines",  desc: "Requirements and process for faculty promotion",    question: "What are the guidelines for promotion?" },
  { icon: "📚", title: "Teaching Load",          desc: "Hours and workload for full-time faculty",          question: "How many hours is a full-time teaching load?" },
  { icon: "🗓️", title: "Leave Benefits",         desc: "Vacation, sick leave, and other policies",         question: "How is leave of absence applied for?" },
  { icon: "⚖️", title: "Tenure Process",         desc: "Understand tenure requirements and evaluation",    question: "What is the tenure process for faculty?" },
  { icon: "📝", title: "Evaluation Criteria",    desc: "Learn how faculty performance is evaluated",       question: "What criteria are used for faculty evaluation?" }
];

// ── Citation Parser ────────────────────────────────────────────────────────────
// Extracts structured citation tags the backend now embeds in context.
// Matches: [Page 12 | Article IV | Section 3 | Some Heading]
function parseCitations(text) {
  if (!text) return { citations: [], cleanText: text };

  const citationRegex = /\[([^\]]{5,120})\]/g;
  const citations = [];
  let match;

  while ((match = citationRegex.exec(text)) !== null) {
    const raw = match[1];
    // Only treat as citation if it contains "Page" or "Article" or "Section"
    if (/page|article|section/i.test(raw)) {
      const parts = raw.split("|").map(p => p.trim());
      const citation = {};
      parts.forEach(part => {
        if (/^page\s+\d+/i.test(part))    citation.page    = part.replace(/^page\s+/i, "");
        else if (/^article\s+/i.test(part)) citation.article = part.replace(/^article\s+/i, "");
        else if (/^section\s+/i.test(part)) citation.section = part.replace(/^section\s+/i, "");
        else                                citation.heading = part;
      });
      citations.push(citation);
    }
  }

  // Remove citation tags from display text
  const cleanText = text.replace(citationRegex, (full, inner) =>
    /page|article|section/i.test(inner) ? "" : full
  ).replace(/\n{3,}/g, "\n\n").trim();

  return { citations, cleanText };
}

// ── Content Formatter ──────────────────────────────────────────────────────────
function formatContent(text) {
  if (!text) return null;

  const { cleanText } = parseCitations(text);
  const lines = cleanText.split("\n");
  const elements = [];
  let listBuffer  = [];
  let numBuffer   = [];
  let key = 0;

  const flushList = () => {
    if (!listBuffer.length) return;
    elements.push(
      <ul key={key++} className="response-list">
        {listBuffer.map((item, i) => (
          <li key={i} className="response-list-item">
            <span className="list-bullet">▸</span>
            <span>{inlineFormat(item)}</span>
          </li>
        ))}
      </ul>
    );
    listBuffer = [];
  };

  const flushNumList = () => {
    if (!numBuffer.length) return;
    elements.push(
      <ol key={key++} className="response-numlist">
        {numBuffer.map((item, i) => (
          <li key={i} className="response-numlist-item">
            <span className="list-num">{i + 1}</span>
            <span>{inlineFormat(item)}</span>
          </li>
        ))}
      </ol>
    );
    numBuffer = [];
  };

  lines.forEach(line => {
    const trimmed = line.trim();
    if (!trimmed) { flushList(); flushNumList(); return; }

    // Numbered list: "1. item" — render as plain bullet (no numbering)
    const numMatch = trimmed.match(/^(\d+)\.\s+(.+)/);
    if (numMatch) { flushNumList(); listBuffer.push(numMatch[2]); return; }

    // Bullet list: "- item", "• item", "* item"
    const bulletMatch = trimmed.match(/^[-•*]\s+(.+)/);
    if (bulletMatch) { flushNumList(); listBuffer.push(bulletMatch[1]); return; }

    // Section header: ALL CAPS or ends with ":"
    const isHeader =
      (/^[A-Z][^a-z]{2,}:?\s*$/.test(trimmed)) ||
      (trimmed.endsWith(":") && trimmed.length < 80 && !trimmed.includes("."));

    if (isHeader) {
      flushList(); flushNumList();
      elements.push(
        <p key={key++} className="response-section-header">
          {trimmed.replace(/:$/, "")}
        </p>
      );
      return;
    }

    // Regular paragraph
    flushList(); flushNumList();
    elements.push(
      <p key={key++} className="response-paragraph">
        {inlineFormat(trimmed)}
      </p>
    );
  });

  flushList();
  flushNumList();
  return elements;
}

// Handles **bold**, *italic*
function inlineFormat(text) {
  const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**"))
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    if (part.startsWith("*") && part.endsWith("*"))
      return <em key={i}>{part.slice(1, -1)}</em>;
    return part;
  });
}

// ── Citation Badge Component ───────────────────────────────────────────────────
function CitationBar({ content }) {
  const { citations } = parseCitations(content || "");

  // Deduplicate citations by page+article
  const unique = citations.filter(
    (c, i, arr) =>
      arr.findIndex(x => x.page === c.page && x.article === c.article) === i
  );

  if (!unique.length) {
    return (
      <div className="metadata-bar">
        <span className="meta-tag source-tag">📖 CPSU FACULTY MANUAL</span>
      </div>
    );
  }

  return (
    <div className="metadata-bar">
      <span className="meta-tag source-tag">📖 CPSU FACULTY MANUAL</span>
      {unique.map((c, i) => (
        <React.Fragment key={i}>
          {c.page    && <span className="meta-tag page-tag">📄 Page {c.page}</span>}
          {c.article && <span className="meta-tag article-tag">§ Article {c.article}</span>}
          {c.section && <span className="meta-tag section-tag">¶ Sec. {c.section}</span>}
        </React.Fragment>
      ))}
    </div>
  );
}

// ── SSE Line Parser ────────────────────────────────────────────────────────────
// Robust: handles partial chunks and multiple events in one read
function parseSSELines(raw) {
  const tokens = [];
  const lines = raw.split("\n");
  for (const line of lines) {
    if (!line.startsWith("data: ")) continue;
    try {
      const data = JSON.parse(line.slice(6).trim());
      if (data.content) tokens.push(data.content);
    } catch {}
  }
  return tokens;
}

// ─────────────────────────────────────────────────────────────────────────────

const Chat = forwardRef(({ sidebarInput, setSidebarInput, setChatHistory, setActiveChatId }, ref) => {
  const [messages,    setMessages]    = useState([]);
  const [input,       setInput]       = useState("");
  const [loading,     setLoading]     = useState(false);
  const [hasMessages, setHasMessages] = useState(false);

  const inputRef          = useRef("");
  const loadingRef        = useRef(false);
  const messagesEndRef    = useRef(null);
  const setChatHistoryRef = useRef(setChatHistory);
  const sessionIdRef      = useRef(crypto.randomUUID());
  const abortCtrlRef      = useRef(null);   // for cancelling in-flight requests

  useEffect(() => { setChatHistoryRef.current = setChatHistory; }, [setChatHistory]);
  useEffect(() => { inputRef.current = input; }, [input]);
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const updateBotMessage = (chatId, accumulated, setMsgs) => {
    setMsgs(prev => {
      const updated = [...prev];
      const lastBot = updated.map(m => m.sender).lastIndexOf("bot");
      if (lastBot !== -1) updated[lastBot] = { ...updated[lastBot], content: accumulated };
      return updated;
    });

    setChatHistoryRef.current(prev =>
      prev.map(c => {
        if (c.id !== chatId) return c;
        const msgs = [...c.messages];
        const lastBot = msgs.map(m => m.sender).lastIndexOf("bot");
        if (lastBot !== -1) msgs[lastBot] = { ...msgs[lastBot], content: accumulated };
        return { ...c, messages: msgs };
      })
    );
  };

  const doSend = async (question) => {
    question = (question ?? inputRef.current).trim();
    if (!question || loadingRef.current) return;

    // Cancel any previous in-flight request
    if (abortCtrlRef.current) abortCtrlRef.current.abort();
    abortCtrlRef.current = new AbortController();

    const userMsg = { sender: "user", text: question };
    const botMsg  = { sender: "bot",  intro: "According to the CPSU Faculty Manual:", content: "" };

    setMessages(prev => [...prev, userMsg, botMsg]);
    setHasMessages(true);
    setLoading(true);
    loadingRef.current = true;
    setInput("");
    inputRef.current = "";
    if (setSidebarInput) setSidebarInput("");

    const chatId = Date.now().toString();
    if (setActiveChatId) setActiveChatId(chatId);

    setChatHistoryRef.current(prev => [
      {
        id: chatId,
        title: question.length > 45 ? question.slice(0, 45) + "…" : question,
        messages: [userMsg, botMsg],
      },
      ...prev,
    ]);

try {
  const response = await fetch(
  "https://characterization-country-leslie-geology.trycloudflare.com/api/chat",
  {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question,
      session_id: sessionIdRef.current,
    }),
    signal: abortCtrlRef.current.signal,
  }
);

  if (!response.ok) {
    throw new Error(`Server error: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  let accumulated = "";
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();

    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });

    // Process complete SSE events
    const parts = buffer.split("\n\n");

    // Keep incomplete data for the next chunk
    buffer = parts.pop() ?? "";

    const tokens = parseSSELines(parts.join("\n\n"));

    if (tokens.length) {
      accumulated += tokens.join("");

      updateBotMessage(
        chatId,
        accumulated,
        setMessages
      );
    }
  }

  // Process remaining buffer
  if (buffer.trim()) {
    const tokens = parseSSELines(buffer);

    if (tokens.length) {
      accumulated += tokens.join("");

      updateBotMessage(
        chatId,
        accumulated,
        setMessages
      );
    }
  }
} catch (err) {
  if (err.name === "AbortError") {
    return;
  }

  const errMsg =
    "❌ Could not connect to KALAW. Please check if the server is running.";

  setMessages(prev => {
    const updated = [...prev];

    const lastBot = updated
      .map(m => m.sender)
      .lastIndexOf("bot");

    if (lastBot !== -1) {
      updated[lastBot] = {
        ...updated[lastBot],
        content: errMsg,
      };
    }

    return updated;
  });
} finally {
  setLoading(false);
  loadingRef.current = false;
}
  };

  const doSendRef = useRef(doSend);
  useEffect(() => { doSendRef.current = doSend; });

  useImperativeHandle(ref, () => ({
    sendQuestion: (q) => doSendRef.current(q),
    newChat: () => {
      // Abort any ongoing stream first
      if (abortCtrlRef.current) abortCtrlRef.current.abort();
      setMessages([]);
      setHasMessages(false);
      setLoading(false);
      loadingRef.current = false;
      setInput("");
      inputRef.current = "";
      sessionIdRef.current = crypto.randomUUID();
      setTimeout(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
        document.querySelector(".chat-input")?.focus();
      }, 50);
    },
    scrollToTop: () => {
      document.querySelector(".messages-container")?.scrollTo({ top: 0, behavior: "smooth" });
    },
    focusInput: () => { document.querySelector(".chat-input")?.focus(); },
    loadChat: (chatId, history) => {
      const found = history.find(c => c.id === chatId);
      if (!found) return;
      setMessages(found.messages);
      setHasMessages(true);
      if (setActiveChatId) setActiveChatId(chatId);
      sessionIdRef.current = chatId;
      setTimeout(() => { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, 50);
    },
  }));

  return (
    <div className="chat-container">
      <div className="messages-container">

        {/* ── Welcome Screen ── */}
        {!hasMessages ? (
          <div className="welcome-screen">
            <div className="welcome-brand">
              <img src={Kalaw} alt="KALAW" className="welcome-logo" />
              <div className="welcome-brand-text">
                <h1 className="welcome-title">KALAW</h1>
                <p className="welcome-tagline">CPSU Faculty Manual Assistant</p>
              </div>
            </div>
            <div className="welcome-divider" />
            <p className="welcome-headline">How can I assist you today?</p>
            <p className="welcome-sub">Select a topic below or type your question to get started.</p>
            <div className="suggestion-grid">
              {SUGGESTIONS.map((s, i) => (
                <button
                  key={i}
                  className="suggestion-card"
                  onClick={() => doSendRef.current(s.question)}
                >
                  <span className="suggestion-icon">{s.icon}</span>
                  <p className="suggestion-title">{s.title}</p>
                  <p className="suggestion-desc">{s.desc}</p>
                </button>
              ))}
            </div>
          </div>

        ) : (
          <>
            {messages.map((msg, i) => (
              <div key={i} className={`message-wrapper ${msg.sender}`}>

                {msg.sender === "bot" ? (
                  <div className="bot-message">

                    {/* ── Dynamic citation bar (reads live from content) ── */}
                    <CitationBar content={msg.content} />

                    {/* ── Intro label ── */}
                    {msg.intro && (
                      <p className="bot-intro-text">{msg.intro}</p>
                    )}

                    {/* ── Formatted answer body ── */}
                    {msg.content ? (
                      <div className="response-body">
                        {formatContent(msg.content)}
                      </div>
                    ) : (
                      /* Show typing indicator while content is empty */
                      !loading && (
                        <p className="response-paragraph empty-response">
                          No response received.
                        </p>
                      )
                    )}

                  </div>
                ) : (
                  <div className="user-query-container">
                    <p className="user-query-text">{msg.text}</p>
                  </div>
                )}

              </div>
            ))}

            {/* ── Loading indicator ── */}
            {loading && (
              <div className="message-wrapper bot">
                <div className="loading-indicator">
                  <div className="dot-pulse">
                    <span /><span /><span />
                  </div>
                  <span>KALAW is reading the manual…</span>
                </div>
              </div>
            )}
          </>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* ── Input Area ── */}
      <div className="input-wrapper">
        <div className="input-area">
          <input
            type="text"
            value={input}
            onChange={e => { setInput(e.target.value); inputRef.current = e.target.value; }}
            onKeyDown={e => e.key === "Enter" && !loading && doSendRef.current()}
            placeholder={loading ? "Waiting for KALAW…" : "Ask a question about faculty policies…"}
            className="chat-input"
            disabled={loading}
          />
          <button
            onClick={() => doSendRef.current()}
            className="chat-send-btn"
            disabled={loading || !input.trim()}
            aria-label="Send question"
          >
            {loading ? "…" : "➤"}
          </button>
        </div>
        <p className="input-footnote">Responses are based on the official CPSU Faculty Manual</p>
      </div>
    </div>
  );
});

export default Chat;