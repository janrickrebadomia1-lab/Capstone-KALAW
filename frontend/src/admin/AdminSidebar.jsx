import React from "react";
import "./Admin.css";

const AdminSidebar = () => {

  return (
    <div className="admin-sidebar">

      <div className="admin-menu">

        <p>📄 Uploaded Documents</p>
        <p>🔄 Rebuild Knowledge</p>
            
      </div>

      <div className="sidebar-footer">
        <small>Admin management panel</small>
      </div>

    </div>
  );

};

export default AdminSidebar;