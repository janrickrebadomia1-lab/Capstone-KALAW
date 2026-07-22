import React from "react";
import Navbar from "../components/Navbar";
import AdminSidebar from "./AdminSidebar";
import UploadPDF from "./UploadPdf";
import "./Admin.css";

const AdminDashboard = () => {
  return (
    <div className="layout-container">

      {/* Same Navbar used by chatbot */}
      <Navbar />

      <div className="app-container">

        {/* Admin Sidebar */}
        <AdminSidebar />

        {/* Main Content */}
        <div className="admin-main-content">

          <h2>Admin Dashboard</h2>
          <p>Manage chatbot knowledge documents.</p>

          <UploadPDF />

        </div>

      </div>

    </div>
  );
};

export default AdminDashboard;