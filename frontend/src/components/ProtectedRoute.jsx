import React from "react";
import { Navigate } from "react-router-dom";

const ProtectedRoute = ({ children }) => {
  const authToken = sessionStorage.getItem("auth-token"); // Or use a context-based auth system

  if (!authToken) {
    return <Navigate to="/login" state={{ message: "Please log in to access all features!" }} />;
  }

  return children;
};

export default ProtectedRoute;