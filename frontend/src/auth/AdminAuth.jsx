import { Navigate } from "react-router-dom";

/**
 * Wrap any admin route that requires login:
 *
 *   <Route path="/admin/manual" element={
 *     <AdminAuth>
 *       <AdminLayout><Admin /></AdminLayout>
 *     </AdminAuth>
 *   } />
 *
 * Checks for the "adminLoggedIn" flag Login.jsx sets in sessionStorage.
 * No token, no backend call — matches the dummy-account login for now.
 */
export default function AdminAuth({ children }) {
  const loggedIn = sessionStorage.getItem("adminLoggedIn") === "true";
  if (!loggedIn) {
    return <Navigate to="/admin/login" replace />;
  }
  return children;
}