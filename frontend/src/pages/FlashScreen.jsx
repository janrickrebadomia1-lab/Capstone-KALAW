import React, { useEffect } from "react";
import Kalaw from "../assets/kalaw.jpg";
import "../styles/FlashScreen.css";

const FlashScreen = ({ onFinish }) => {
  useEffect(() => {
    const timer = setTimeout(onFinish, 5500);
    return () => clearTimeout(timer);
  }, [onFinish]);

  return (
    <div className="flash-screen">
      <img src={Kalaw} alt="KALAW" className="flash-logo" />
      <h1 className="flash-title">Welcome to KALAW</h1>
      <p className="flash-subtitle">Your CPSU Faculty Manual Assistant</p>
    </div>
  );
};

export default FlashScreen;