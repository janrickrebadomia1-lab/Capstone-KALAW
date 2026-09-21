import { useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import logo from "../assets/logo.png"; 
import "../styles/AdminLayout.css";

const NAV_ITEMS = [
  { 
    label: "Faculty Manual", 
    to: "/admin/manual", 
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
        <polyline points="14 2 14 8 20 8" />
        <line x1="16" y1="13" x2="8" y2="13" />
        <line x1="16" y1="17" x2="8" y2="17" />
      </svg>
    ) 
  },
];

export default function AdminLayout({ children }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const navigate = useNavigate();

  function handleLogout() {
    sessionStorage.removeItem("adminLoggedIn");
    navigate("/admin/login");
  }

  return (
    <div className="adm-shell">
      {/* Mobile Top Header Bar */}
      <header className="adm-mobile-header">
        <div className="adm-brand-compact">
          <div className="adm-brand-logo">
            <img src={logo} alt="CPSU Logo" className="adm-brand-logo-img" />
          </div>
          <span className="adm-brand-name">CPSU FacultyGuide Admin</span>
        </div>
        <button 
          className="adm-hamburger" 
          onClick={() => setMobileOpen(!mobileOpen)}
          aria-label="Toggle menu"
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            {mobileOpen ? (
              <path d="M18 6L6 18M6 6l12 12" />
            ) : (
              <path d="M4 6h16M4 12h16M4 18h16" />
            )}
          </svg>
        </button>
      </header>

      {/* Backdrop for Mobile Menu */}
      {mobileOpen && (
        <div className="adm-backdrop" onClick={() => setMobileOpen(false)} />
      )}

      {/* Main Sidebar */}
      <aside className={`adm-sidebar ${mobileOpen ? "adm-sidebar-open" : ""}`}>
        <div className="adm-brand">
          <div className="adm-brand-mark">
            <img src={logo} alt="CPSU Logo" className="adm-brand-logo-img" />
          </div>
          <div className="adm-brand-details">
            <span className="adm-brand-title">CPSU FacultyGuide</span>
            <span className="adm-brand-sub">Admin</span>
          </div>
        </div>

        <nav className="adm-nav">
          <div className="adm-nav-header">Manage</div>
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.label}
              to={item.to}
              onClick={() => setMobileOpen(false)}
              className={({ isActive }) => 
                `adm-nav-item ${isActive ? "adm-nav-active" : ""}`
              }
            >
              <span className="adm-nav-icon">{item.icon}</span>
              <span className="adm-nav-label">{item.label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="adm-sidebar-footer">
          <button className="adm-back-btn adm-logout-btn" onClick={handleLogout} type="button">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
              <polyline points="16 17 21 12 16 7" />
              <line x1="21" y1="12" x2="9" y2="12" />
            </svg>
            <span>Log Out</span>
          </button>
          <NavLink to="/" className="adm-back-btn">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="19" y1="12" x2="5" y2="12" />
              <polyline points="12 19 5 12 12 5" />
            </svg>
            <span>Back to Chatbot</span>
          </NavLink>
        </div>
      </aside>

      {/* Scrollable Main Workspace */}
      <main className="adm-content">{children}</main>
    </div>
  );
}