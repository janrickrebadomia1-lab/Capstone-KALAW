import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout";
import FlashScreen from "./pages/FlashScreen";
import Admin from "./admin/Admin";
import AdminLayout from "./admin/AdminLayout";
import AdminAuth from "./auth/AdminAuth";
import Login from "./auth/Login";

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
        <Route path="/admin" element={<Navigate to="/admin/manual" replace />} />
        <Route path="/admin/login" element={<Login />} />
        <Route path="/admin/manual" element={<AdminAuth><AdminLayout><Admin /></AdminLayout> </AdminAuth>} />
      </Routes>
    </Router>
  );
}

export default App;