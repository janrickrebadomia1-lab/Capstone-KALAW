import { useState } from "react";
import { useNavigate } from "react-router-dom";
import logo from "../assets/logo.png";
import "../styles/Login.css";

/**
 * Dummy admin login — checks against a hardcoded username/password
 * right here in the frontend. No backend call, no real security.
 * This is a placeholder gate, not authentication — anyone who reads
 * the JS bundle can see these values. Swap this for a real backend
 * check before this goes anywhere other admins/students could reach.
 */
const DUMMY_USERNAME = "admin";
const DUMMY_PASSWORD = "CPSU";

export default function Login() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  function handleSubmit(e) {
    e.preventDefault();
    if (username === DUMMY_USERNAME && password === DUMMY_PASSWORD) {
      sessionStorage.setItem("adminLoggedIn", "true");
      navigate("/admin/manual");
    } else {
      setError("Incorrect username or password.");
    }
  }

  return (
    <div className="log-wrap">
      <form className="log-card" onSubmit={handleSubmit}>
        <div className="log-brand">
          <img src={logo} alt="Kalaw" className="log-brand-logo" />
          <h1 className="log-brand-title">KALAW</h1>
          <p className="log-brand-tagline">CPSU Faculty Manual Assistant</p>
          <div className="log-divider" />
          <span className="log-pill">Admin Access</span>
        </div>

        {error && (
          <div className="log-alert">
            <span className="log-alert-icon">✕</span>
            <span>{error}</span>
          </div>
        )}

        <div className="log-field">
          <label className="log-label" htmlFor="adminUsername">Username</label>
          <div className="log-input-wrap">
            <svg className="log-input-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
              <circle cx="12" cy="7" r="4" />
            </svg>
            <input
              id="adminUsername"
              className="log-input"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter username"
              autoFocus
              autoComplete="username"
            />
          </div>
        </div>

        <div className="log-field">
          <label className="log-label" htmlFor="adminPassword">Password</label>
          <div className="log-input-wrap">
            <svg className="log-input-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="11" width="18" height="11" rx="2" />
              <path d="M7 11V7a5 5 0 0 1 10 0v4" />
            </svg>
            <input
              id="adminPassword"
              className="log-input"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter password"
              autoComplete="current-password"
            />
          </div>
        </div>

        <button className="log-btn-primary" type="submit" disabled={!username || !password}>
          Sign In
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <line x1="5" y1="12" x2="19" y2="12" />
            <polyline points="12 5 19 12 12 19" />
          </svg>
        </button>

        <p className="log-footnote">Faculty Manual retrieval index · internal admin tool</p>
      </form>
    </div>
  );
}