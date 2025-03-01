import './App.css';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import Home from './pages/Home';
import Dashboard from './pages/Dashboard';
import { useState, useEffect } from 'react';
import Alert from './components/Alert';
import Navbar from './components/Navbar';
import Login from './pages/Login';
import Signup from './pages/Signup';

function App() {
  const location = useLocation();
  const [alert, setAlert] = useState(null);
  const [darkMode, setDarkMode] = useState(false);
  const [fontSize, setFontSize] = useState('medium');

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
          <Route exact path="/" element={<Home showAlert={showAlert} darkMode={darkMode} />} />
          {/* Login */}
          <Route exact path="/login" element={<Login showAlert={showAlert} darkMode={darkMode} />} />
          {/* Signup */}
          <Route exact path="/signup" element={<Signup showAlert={showAlert} darkMode={darkMode} />} />
          {/* Dashboard */}
          <Route exact path="/dashboard" element={
            <Dashboard 
              showAlert={showAlert} 
              darkMode={darkMode} 
              setDarkMode={setDarkMode}
              fontSize={fontSize}
              setFontSize={setFontSize}
            />
          } />
        </Routes>
      </div>
    </div>
  );
}

function AppWithRouter() {
  return (
    <Router>
      <App />
    </Router>
  );
}

export default AppWithRouter;