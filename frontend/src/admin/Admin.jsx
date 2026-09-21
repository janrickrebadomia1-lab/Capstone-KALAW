import { useState, useRef, useCallback } from "react";
import "../styles/Admin.css";

const API_BASE = import.meta.env.VITE_API_BASE || "/api";

const PIPELINE_STEPS = [
  "Uploading Document PDF",
  "Extracting Raw Text",
  "Chunking Manual Sections",
  "Generating Vector Embeddings",
  "Rebuilding Search Index",
];

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function Admin() {
  const [status, setStatus] = useState({
    filename: "CPSU-Faculty-Manual.pdf",
    updated: "—",
    chunks: "—",
  });
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [stepIndex, setStepIndex] = useState(-1);
  const [banner, setBanner] = useState(null);
  const [history, setHistory] = useState([]);
  const [token, setToken] = useState(import.meta.env.VITE_ADMIN_TOKEN || "");
  const inputRef = useRef(null);

  const running = stepIndex >= 0 && stepIndex < PIPELINE_STEPS.length;

  const handleFile = useCallback((f) => {
    if (!f) return;
    if (f.type !== "application/pdf" && !f.name.toLowerCase().endsWith(".pdf")) {
      setBanner({ type: "err", msg: "Invalid file format. Only PDF documents are permitted." });
      return;
    }
    setFile(f);
    setBanner(null);
  }, []);

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files?.[0]);
  };

  async function fetchStatus() {
    try {
      const res = await fetch(`${API_BASE}/admin/manual-status`, {
        headers: { "X-Admin-Token": token },
      });
      if (!res.ok) return;
      const data = await res.json();
      setStatus({
        filename: data.pdf_present ? "CPSU-Faculty-Manual.pdf" : "No manual uploaded",
        updated: data.pdf_last_modified || "—",
        chunks: data.chunks_indexed?.toLocaleString?.() ?? data.chunks_indexed ?? "—",
      });
    } catch {
      // Keep static initial states on network error
    }
  }

  async function runUpload() {
    if (!file) return;
    setBanner(null);
    setStepIndex(0);

    const form = new FormData();
    form.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/admin/upload-manual`, {
        method: "POST",
        headers: { "X-Admin-Token": token },
        body: form,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Upload task rejected by backend");
      }
      const data = await res.json();
      setStepIndex(PIPELINE_STEPS.length);
      finishSuccess(data.chunks, data.filename);
    } catch (err) {
      setStepIndex(-1);
      setBanner({ type: "err", msg: err.message || "Failed to parse manual. Previous index retained." });
      setHistory((h) => [
        { name: file.name, time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }), ok: false },
        ...h,
      ]);
    }
  }

  function finishSuccess(chunks, filename) {
    const chunkStr = typeof chunks === "number" ? chunks.toLocaleString() : chunks;
    setStatus({ filename: filename || file.name, updated: "Just now", chunks: chunkStr });
    setBanner({ type: "ok", msg: `Indexing complete. ${chunkStr} text chunks processed successfully.` });
    setHistory((h) => [
      { name: filename || file.name, time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }), chunks: chunkStr, ok: true },
      ...h,
    ]);
    setTimeout(() => {
      setStepIndex(-1);
      setFile(null);
      if (inputRef.current) inputRef.current.value = "";
    }, 1200);
  }

  return (
    <div className="amu-container">
      {/* Page Header Area */}
      <header className="amu-header">
        <div>
          <h1 className="amu-title">Faculty Manual Configuration</h1>
          <p className="amu-subtitle">
            Manage institutional knowledge indexes and maintain operational vector embeddings.
          </p>
        </div>
      </header>

      {/* Overview Metrics Cards */}
      <div className="amu-grid-2">
        <div className="amu-card">
          <div className="amu-card-header">
            <span className="amu-card-label">Active Document</span>
            <span className="amu-badge-live">Live Index</span>
          </div>
          <div className="amu-card-value">{status.filename}</div>
          <div className="amu-card-footer">Last updated {status.updated}</div>
        </div>

        <div className="amu-card">
          <div className="amu-card-header">
            <span className="amu-card-label">Indexed Vector Chunks</span>
            <button className="amu-text-action" onClick={fetchStatus} type="button">
              Refresh Status
            </button>
          </div>
          <div className="amu-card-value amu-mono">{status.chunks}</div>
          <div className="amu-card-footer">Active query search units</div>
        </div>
      </div>

      {/* System Notification Banner */}
      {banner && (
        <div className={`amu-alert amu-alert-${banner.type}`}>
          <div className="amu-alert-icon">
            {banner.type === "ok" ? "✓" : banner.type === "err" ? "✕" : "ℹ"}
          </div>
          <div className="amu-alert-text">{banner.msg}</div>
        </div>
      )}

      {/* Main Form Section */}
      <section className="amu-section">
        <h2 className="amu-section-title">Upload New Document</h2>

        {!file ? (
          <label
            className={`amu-dropzone ${dragging ? "amu-dropzone-active" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            <input
              ref={inputRef}
              type="file"
              accept="application/pdf"
              onChange={(e) => handleFile(e.target.files?.[0])}
            />
            <div className="amu-dz-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>
            <p className="amu-dz-title">Click to upload or drag and drop file</p>
            <p className="amu-dz-hint">Standard PDF formats accepted (Max file size: 50MB)</p>
          </label>
        ) : (
          <div className="amu-file-box">
            <div className="amu-file-info">
              <div className="amu-file-icon">PDF</div>
              <div>
                <div className="amu-file-name">{file.name}</div>
                <div className="amu-file-meta">{formatBytes(file.size)}</div>
              </div>
            </div>
            <button
              className="amu-btn-ghost"
              type="button"
              onClick={() => { setFile(null); if (inputRef.current) inputRef.current.value = ""; }}
            >
              Remove
            </button>
          </div>
        )}

        <button 
          className="amu-btn-primary" 
          disabled={!file || running} 
          onClick={runUpload} 
          type="button"
        >
          {running ? "Processing Pipeline..." : "Process and Rebuild Index"}
        </button>

        {/* Processing State */}
        {running && (
          <div className="amu-pipeline">
            <h3 className="amu-pipeline-heading">Execution Status</h3>
            <div className="amu-steps">
              {PIPELINE_STEPS.map((label, i) => {
                const state = i < stepIndex ? "completed" : i === stepIndex ? "active" : "pending";
                return (
                  <div key={label} className={`amu-step amu-step-${state}`}>
                    <div className="amu-step-indicator">
                      {state === "completed" ? (
                        "✓"
                      ) : state === "active" ? (
                        <div className="amu-spinner" />
                      ) : (
                        i + 1
                      )}
                    </div>
                    <span className="amu-step-label">{label}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </section>

      {/* Execution Log Table */}
      <section className="amu-section">
        <h2 className="amu-section-title">Upload Audit History</h2>
        <div className="amu-table-wrapper">
          <table className="amu-table">
            <thead>
              <tr>
                <th>Document Name</th>
                <th>Time Executed</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {history.length === 0 ? (
                <tr>
                  <td colSpan={3} className="amu-table-empty">
                    No upload events recorded for current session.
                  </td>
                </tr>
              ) : (
                history.map((h, i) => (
                  <tr key={i}>
                    <td className="amu-font-medium">{h.name}</td>
                    <td className="amu-text-sub">{h.time}</td>
                    <td>
                      <span className={`amu-pill ${h.ok ? "amu-pill-success" : "amu-pill-error"}`}>
                        {h.ok ? `${h.chunks} Chunks` : "Failed"}
                      </span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </section>

      {/* Advanced Connection Accordion */}
      <details className="amu-accordion">
        <summary className="amu-accordion-trigger">
          <span>Backend Configuration</span>
          <svg className="amu-chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </summary>
        <div className="amu-accordion-content">
          <div className="amu-field">
            <label className="amu-label">API Base URL</label>
            <input className="amu-input" type="text" value={API_BASE} disabled />
            <span className="amu-field-hint">Controlled via Environment variable (VITE_API_BASE)</span>
          </div>
          <div className="amu-field">
            <label className="amu-label">Admin Authentication Token</label>
            <input
              className="amu-input"
              type="password"
              value={token}
              onChange={(e) => setToken(e.target.value)}
              placeholder="Enter authorization token"
            />
          </div>
        </div>
      </details>
    </div>
  );
}