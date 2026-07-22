import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import FlashScreen from "./pages/FlashScreen";
import AdminDashboard from "./admin/AdminDashboard";

function ChatWithFlash() {
  const [showFlash, setShowFlash] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => setShowFlash(false), 3000);
    return () => clearTimeout(timer);
  }, []);

  if (showFlash) return <FlashScreen />;
  return <Layout />;
}

function App() {
  return (
    <Router>
      <Routes>

        <Route path="/" element={<ChatWithFlash />} />

        <Route path="/admin" element={<AdminDashboard />} />

      </Routes>
    </Router>
  );
}

export default App;