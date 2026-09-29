import { Navigate } from "react-router-dom";

export default function AdminAuth({ children }) {
  const loggedIn = sessionStorage.getItem("adminLoggedIn") === "true";
  if (!loggedIn) {
    return <Navigate to="/admin/login" replace />;
  }
  return children;
}