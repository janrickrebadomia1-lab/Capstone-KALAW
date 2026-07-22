import React from "react";
import "../styles/Layout.css";
import Logo from "../assets/logo.png";

const Navbar = ({ onHamburgerClick }) => {
  return (
    <div className="navbar">

      <div className="navbar-left">
        {/* Hamburger — only visible on mobile via CSS */}
        <button
          className="hamburger-btn"
          onClick={onHamburgerClick}
          aria-label="Toggle sidebar"
        >
          ☰
        </button>

        <img src={Logo} alt="CPSU Logo" className="logo-img" />
        <div className="navbar-title">
          <span className="logo">CPSU FacultyGuide</span>
        </div>
      </div>

      <div className="navbar-container">
        <div className="navbar-right">
          <span className="welcome">Welcome</span>
        </div>
      </div>

    </div>
  );
};

export default Navbar;