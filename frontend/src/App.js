import './App.css';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import { useState, useEffect } from 'react';
import Home from './pages/Home/Home';
import Dashboard from './pages/Dashboard/Dashboard';
import Alert from './components/Alert';
import Navbar from './components/Navbar/Navbar';
import Login from './pages/Login/Login';
import Signup from './pages/Signup/Signup';
import ProtectedRoute from './components/ProtectedRoute';
import About from './pages/About/About';
import QueryHistory from './pages/QueryHistory/QueryHistory';
import UserProfile from './pages/UserProflie/UserProfile';
import CustomerManagement from './pages/customer_manager/CustomerManagement';

function App() {
  const location = useLocation();
  const [alert, setAlert] = useState(null);
  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem('darkMode')==='true'; // or your default
  });
  const [fontSize, setFontSize] = useState(() => {
    return localStorage.getItem('fontSize') || 'small'; // or your default
  });
  

  // Load darkMode preference from localStorage on initial load
  useEffect(() => {
    const savedDarkMode = localStorage.getItem('darkMode');
    if (savedDarkMode !== null) {
      setDarkMode(savedDarkMode === 'true');
    }

    const savedFontSize = localStorage.getItem('fontSize');
    if (savedFontSize) {
      setFontSize(savedFontSize);
    }
  }, []);

  const showAlert = (message, type) => {
    setAlert({
      msg: message,
      type: type
    });
    setTimeout(() => {
      setAlert(null);
    }, 1500);
  };

  return (
    <div className={darkMode ? "app dark-mode" : "app"} style={{ minHeight: '100vh', width: '100%', margin: 0, padding: 0 }}>
      {(location.pathname !== '/login' && location.pathname !== '/signup') &&
        <Navbar darkMode={darkMode} setDarkMode={setDarkMode} />}
      <Alert alert={alert} />
      <div className='container'>
        <Routes>
          {/* Home */}
          <Route exact path="/" element={
            <Home showAlert={showAlert} darkMode={darkMode} />} />
          {/* About */}
          <Route exact path="/about" element={
            <About showAlert={showAlert} darkMode={darkMode} />} />
          {/* Login */}
          <Route exact path="/login" element={
            <Login showAlert={showAlert} darkMode={darkMode} />} />
          {/* Signup */}
          <Route exact path="/signup" element={
            <Signup showAlert={showAlert} darkMode={darkMode} />} />
          {/* Dashboard */}
          <Route exact path="/dashboard" element={
            <ProtectedRoute>
              <Dashboard
                showAlert={showAlert}
                darkMode={darkMode}
                setDarkMode={setDarkMode}
                fontSize={fontSize}
                setFontSize={setFontSize}
              />
            </ProtectedRoute>
          } />
          {/* QueryHistory */}
          <Route exact path="/history" element={
            <ProtectedRoute>
              <QueryHistory showAlert={showAlert} darkMode={darkMode} />
            </ProtectedRoute>
          } />
          {/* UserProfile */}
          <Route exact path="/profile" element={
            <ProtectedRoute>
              <UserProfile showAlert={showAlert} darkMode={darkMode} />
            </ProtectedRoute>
          } />
          <Route exact path="/dashboard/customer" element={
            <ProtectedRoute>
              <CustomerManagement showAlert={showAlert} darkMode={darkMode} />
            </ProtectedRoute>
          } />
        </Routes>
      </div>
    </div>
  );
}



export default function AppWithRouter() {
  return (
    <Router>
      <App />
    </Router>
  );
};