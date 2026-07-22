import React, { useState } from "react";
import "./Admin.css";

const UploadPDF = () => {

  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");

  const handleUpload = () => {

    if (!file) {
      setMessage("Please select a PDF file.");
      return;
    }

    setMessage("PDF uploaded successfully.");

  };

  return (

    <div className="admin-card">

      <h3>Upload Faculty Manual</h3>

      <input
        type="file"
        accept="application/pdf"
        onChange={(e) => setFile(e.target.files[0])}
      />

      <button className="upload-btn" onClick={handleUpload}>
        Upload
      </button>

      <p>{message}</p>

    </div>

  );

};

export default UploadPDF;