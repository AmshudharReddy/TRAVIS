import React, { useState, useEffect } from "react";
import "./Dashboard.css";
import { FaPaperPlane } from "react-icons/fa"; // Import paper plane icon for submit button

const Dashboard = ({ darkMode, setDarkMode, fontSize, setFontSize, showAlert }) => {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState(null);
  const [activePopup, setActivePopup] = useState(null);
  const [notificationsEnabled, setNotificationsEnabled] = useState(false);
  const [autoReadEnabled, setAutoReadEnabled] = useState(false);

  const handleQuerySubmit = () => {
    if (query.trim() === "") return;
    // Placeholder for AI processing logic
    setResponse(`Response for: "${query}"`);
    setQuery("");
    setActivePopup(null);
  };

  const togglePopup = (popupName) => {
    if (activePopup === popupName) {
      setActivePopup(null);
    } else {
      setActivePopup(popupName);
    }
  };

  // Save font size changes to localStorage
  useEffect(() => {
    localStorage.setItem('fontSize', fontSize);
  }, [fontSize]);

  // Handle Enter key press
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleQuerySubmit();
    }
  };

  // Recent highlights/history items
  const recentHighlights = [
    { id: 1, query: "Sample Query 1", response: "Sample Response 1" },
    { id: 2, query: "Sample Query 2", response: "Sample Response 2" },
    { id: 3, query: "Sample Query 3", response: "Sample Response 3" },
  ];

  return (
    <div className={`dashboard ${darkMode ? "dark-mode" : ""}`}>
      <div className="dashboard-inner">
        {/* Main Content */}
        <main className="main-content">
          {/* Welcome Message - Now positioned at the top */}
          <div className="welcome-message">
            <h2>👋 Welcome to TRAVIS AI Assistant!</h2>
            <p>Your intelligent companion for all your questions and tasks. Type a query below or use the tools to interact.</p>
          </div>

          {/* Query Input Section - Moved below welcome message */}
          <div className="query-section">
            <div className="query-input-container">
              <textarea
                className="query-input"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyPress}
                placeholder="Type your query here..."
                style={{ fontSize: `var(--font-size-${fontSize})` }}
              />
              <button 
                onClick={handleQuerySubmit} 
                className="process-btn"
                title="Process Query"
                disabled={query.trim() === ""}
              >
                <FaPaperPlane />
              </button>
            </div>
          </div>
          
          {/* Button Navigation */}
          <div className="button-nav">
            <button 
              className="icon-btn" 
              onClick={() => togglePopup('speak')}
              title="Speak"
            >
              <div className="circle-icon">🎤</div>
              <span>Speak</span>
            </button>
            
            <button 
              className="icon-btn" 
              onClick={() => togglePopup('recent')}
              title="Recent Highlights"
            >
              <div className="circle-icon">📊</div>
              <span>Recent</span>
            </button>
            
            <button 
              className="icon-btn" 
              onClick={() => togglePopup('fontSize')}
              title="Font Size"
            >
              <div className="circle-icon">Aa</div>
              <span>Font Size</span>
            </button>
            
            <button 
              className="icon-btn" 
              onClick={() => togglePopup('settings')}
              title="Settings"
            >
              <div className="circle-icon">⚙️</div>
              <span>Settings</span>
            </button>
          </div>

          {/* Response Display */}
          {response && (
            <section className="response-section">
              <h3>AI Response:</h3>
              <div className="response-content" style={{ fontSize: `var(--font-size-${fontSize})` }}>
                <p>{response}</p>
                <button className="tts-btn">🔊 Listen</button>
              </div>
            </section>
          )}
        </main>

        {/* Popups */}
        {activePopup && (
          <div className="popup-overlay" onClick={() => setActivePopup(null)}>
            <div className="popup-container" onClick={(e) => e.stopPropagation()}>
              {/* Speak Popup */}
              {activePopup === 'speak' && (
                <div className="popup-content">
                  <h2>Voice Input</h2>
                  <div className="voice-input-container">
                    <div className="voice-icon">🎤</div>
                    <p>Click to start speaking...</p>
                  </div>
                  <div className="popup-actions">
                    <button className="primary-btn">Start Recording</button>
                    <button onClick={() => setActivePopup(null)} className="secondary-btn">Cancel</button>
                  </div>
                </div>
              )}

              {/* Recent Highlights Popup */}
              {activePopup === 'recent' && (
                <div className="popup-content">
                  <h2>Recent Highlights</h2>
                  <ul className="highlights-list">
                    {recentHighlights.map((item) => (
                      <li key={item.id} className="highlight-item">
                        <div className="highlight-query">{item.query}</div>
                        <div className="highlight-response">{item.response}</div>
                      </li>
                    ))}
                  </ul>
                  <div className="popup-actions">
                    <button onClick={() => setActivePopup(null)} className="secondary-btn">Close</button>
                  </div>
                </div>
              )}

              {/* Font Size Popup */}
              {activePopup === 'fontSize' && (
                <div className="popup-content">
                  <h2>Font Size</h2>
                  <div className="font-size-options">
                    <button 
                      className={`font-option ${fontSize === 'small' ? 'active' : ''}`}
                      onClick={() => setFontSize('small')}
                    >
                      Small
                    </button>
                    <button 
                      className={`font-option ${fontSize === 'medium' ? 'active' : ''}`}
                      onClick={() => setFontSize('medium')}
                    >
                      Medium
                    </button>
                    <button 
                      className={`font-option ${fontSize === 'large' ? 'active' : ''}`}
                      onClick={() => setFontSize('large')}
                    >
                      Large
                    </button>
                  </div>
                  <div className="popup-actions">
                    <button onClick={() => setActivePopup(null)} className="secondary-btn">Close</button>
                  </div>
                </div>
              )}

              {/* Settings Popup */}
              {activePopup === 'settings' && (
                <div className="popup-content">
                  <h2>Settings</h2>
                  <div className="settings-list">
                    <div className="setting-item">
                      <label>
                        <span>Dark Mode</span>
                        <button 
                          className={`toggle-switch ${darkMode ? 'active' : ''}`}
                          onClick={() => {
                            setDarkMode(!darkMode);
                            localStorage.setItem('darkMode', !darkMode);
                          }}
                        >
                          {darkMode ? "ON" : "OFF"}
                        </button>
                      </label>
                    </div>
                    <div className="setting-item">
                      <label>
                        <span>Notifications</span>
                        <button 
                          className={`toggle-switch ${notificationsEnabled ? 'active' : ''}`}
                          onClick={() => setNotificationsEnabled(!notificationsEnabled)}
                        >
                          {notificationsEnabled ? "ON" : "OFF"}
                        </button>
                      </label>
                    </div>
                    <div className="setting-item">
                      <label>
                        <span>Auto-read Responses</span>
                        <button 
                          className={`toggle-switch ${autoReadEnabled ? 'active' : ''}`}
                          onClick={() => setAutoReadEnabled(!autoReadEnabled)}
                        >
                          {autoReadEnabled ? "ON" : "OFF"}
                        </button>
                      </label>
                    </div>
                  </div>
                  <div className="popup-actions">
                    <button onClick={() => setActivePopup(null)} className="secondary-btn">Close</button>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Footer */}
        <footer className="footer">
          <p>© 2025 TRAVIS AI Assistant | v2.0.3</p>
        </footer>
      </div>
    </div>
  );
};

export default Dashboard;