import React, { useState, useEffect, useRef } from "react";
import { Link } from 'react-router-dom';
import "./Dashboard.css";
import { FaPaperPlane, FaTimes, FaVolumeUp, FaMicrophone, FaLanguage } from "react-icons/fa";

const Dashboard = ({ darkMode, setDarkMode, fontSize, setFontSize, showAlert }) => {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState(null);
  const [translatedResponse, setTranslatedResponse] = useState(null);
  const [responseCategory, setResponseCategory] = useState(null);
  const [lastQuery, setLastQuery] = useState("");
  const [activePopup, setActivePopup] = useState(null);
  const [notificationsEnabled, setNotificationsEnabled] = useState(false);
  const [autoReadEnabled, setAutoReadEnabled] = useState(false);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [isTranslating, setIsTranslating] = useState(false);
  const responseRef = useRef(null);
  const responseContentRef = useRef(null);
  const conversationContainerRef = useRef(null);

  const dashboardClasses = `dashboard ${darkMode ? "dark-mode" : ""} ${response ? "response-active" : ""} font-size-${fontSize}`;

  const speak = (text) => {
    if ("speechSynthesis" in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = "en-US";
      utterance.rate = 1;
      window.speechSynthesis.speak(utterance);
      console.log("Voice output");
    } else {
      alert("Text-to-Speech not supported in this browser.");
    }
  };

  const handleQuerySubmit = async (e) => {
    if (e) e.preventDefault();

    if (!query.trim()) return;

    const currentQuery = query.trim();
    const authToken = sessionStorage.getItem("auth-token");

    if (!authToken) {
      console.error("User is not authenticated");
      return;
    }

    setLastQuery(currentQuery);
    setQuery("");
    setTranslatedResponse(null); // Reset translation when submitting a new query

    try {
      const response = await fetch("http://localhost:5000/api/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "auth-token": authToken,
        },
        body: JSON.stringify({ query: currentQuery }),
      });

      const data = await response.json();
      if (response.ok) {
        console.log("Query processed:", data);
        setResponse(data.response || "No response received");
        setResponseCategory(data.category || "General");
        fetchQueryHistory();
        if (autoReadEnabled) {
          speak(data.response);
        }
        
        // Removed auto-scrolling behavior here
        
      } else {
        console.error("Error:", data.error);
        setResponse(data.error || "An error occurred while processing the query");
        setResponseCategory("Error");
      }
    } catch (error) {
      console.error("Request failed:", error);
      setResponse("Network error. Please try again.");
      setResponseCategory("Error");
    }
  };

  const handleTranslate = async () => {
    if (!response) return;
    
    setIsTranslating(true);

    const authToken = sessionStorage.getItem("auth-token");
    if (!authToken) {
      console.error("User is not authenticated");
      setIsTranslating(false);
      return;
    }

    try {
      const translatedResponse = await fetch("http://localhost:5000/api/query/translate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "auth-token": authToken,
        },
        body: JSON.stringify({ response: response }),
      });

      const data = await translatedResponse.json();
      if (translatedResponse.ok) {
        console.log("Translated Response", data);
        setTranslatedResponse(data.translation || "No translation received");
        
        // Removed auto-scrolling after translation
      } else {
        setTranslatedResponse("Translation error: " + (data.error || "Unknown error"));
      }
    } catch (error) {
      console.error("Translation request failed:", error);
      setTranslatedResponse("Network error. Please try again.");
    } finally {
      setIsTranslating(false);
    }
  };


  const handleCloseResponse = () => {
    setIsTransitioning(true);
    setTimeout(() => {
      setResponse(null);
      setTranslatedResponse(null);
      setResponseCategory(null);
      setLastQuery("");
      setIsTransitioning(false);
      setQuery("");
    }, 300);
  };

  const togglePopup = (popupName) => {
    if (activePopup === popupName) {
      setActivePopup(null);
    } else {
      setActivePopup(popupName);
    }
  };

  useEffect(() => {
    localStorage.setItem('fontSize', fontSize);
  }, [fontSize]);

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleQuerySubmit();
    }
  };

  const [queryHistory, setQueryHistory] = useState([]);
  const [recentHighlights, setRecentHighlights] = useState([]);

  const fetchQueryHistory = () => {
    const authToken = sessionStorage.getItem("auth-token");

    if (!authToken) {
      console.error("User is not authenticated");
      return;
    }

    fetch("http://localhost:5000/api/query/history?limit=3&sort=-createdAt", {
      headers: { "auth-token": authToken }
    })
      .then((res) => res.json())
      .then((data) => {
        setQueryHistory(data);
        const highlights = data
          .slice(0, 3)
          .map((entry, index) => ({
            id: index + 1,
            query: entry.query,
            response: entry.response,
            category: entry.category || "General"
          }));
        setRecentHighlights(highlights);
      })
      .catch((err) => {
        console.error("Error fetching history:", err);
      });
  };

  // Removed the useEffect that automatically scrolled after response or translation

  useEffect(() => {
    fetchQueryHistory();
  }, []);

  return (
    <div className={dashboardClasses}>
      <div className="dashboard-inner">
        {/* Conversation container with scrolling */}
        <div className="conversation-container" ref={conversationContainerRef}>
          <main className="main-content">
            {!response && (
              <div className="welcome-message">
                <h2>👋 Welcome to TRAVIS AI Assistant!</h2>
                <p>Your intelligent companion for all your questions and tasks. Simply type a query below or use voice commands to get started. Our advanced AI is ready to assist you with information, creative tasks, and problem-solving.</p>
              </div>
            )}

            {response && (
              <div className="response-mode" ref={responseRef}>
                <div className="response-header">
                  <div className="query-display">
                    <h3>{lastQuery}</h3>
                    {responseCategory && <span className="response-category">{responseCategory}</span>}
                  </div>
                  <button onClick={handleCloseResponse} className="close-response-btn" title="Close Response">
                    <FaTimes />
                  </button>
                </div>
                <div className="response-content" ref={responseContentRef}>
                  <p><strong>Response:</strong> {response}</p>
                  
                  {translatedResponse && (
                    <div className="translated-content">
                      <p><strong>Telugu Translation:</strong> {translatedResponse}</p>
                    </div>
                  )}
                  
                  <div className="response-actions">
                    <button 
                      onClick={() => speak(response)} 
                      className="tts-btn" 
                      title="Listen to response"
                    >
                      <FaVolumeUp />{' '}Listen
                    </button>
                    
                    <button 
                      onClick={handleTranslate} 
                      className="translate-btn" 
                      title="Translate Response"
                      disabled={isTranslating}
                    >
                      {isTranslating ? "Translating..." : <><FaLanguage />{'   '}Translate</>}
                    </button>

                  </div>
                </div>
              </div>
            )}

            {!response && (
              <div className="query-label">
                <p style={{ fontSize: "20px", textAlign: "center" }}>What's your query?</p>
              </div>
            )}

            {!response && (
              <div className="query-section">
                <div className="query-input-container">
                  <textarea
                    className="query-input"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={handleKeyPress}
                    placeholder="Ask me anything..."
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
            )}

            {!response && (
              <div className="button-nav">
                <button className="icon-btn" onClick={() => togglePopup('speak')} title="Speak">
                  <div className="circle-icon">🎤</div>
                  <span>Speak</span>
                </button>
                <button className="icon-btn" onClick={() => togglePopup('recent')} title="Recent Highlights">
                  <div className="circle-icon">📊</div>
                  <span>Recent</span>
                </button>
                <button className="icon-btn" onClick={() => togglePopup('fontSize')} title="Font Size">
                  <div className="circle-icon">Aa</div>
                  <span>Font Size</span>
                </button>
                <button className="icon-btn" onClick={() => togglePopup('settings')} title="Settings">
                  <div className="circle-icon">⚙️</div>
                  <span>Settings</span>
                </button>
              </div>
            )}
            
            {!response && (
              <footer className="footer">
                <p>© 2025 TRAVIS AI Assistant | v2.0.3</p>
              </footer>
            )}
            
            {/* Add spacer div when in response mode to ensure proper padding at bottom */}
            {response && <div className="response-bottom-spacer"></div>}
          </main>
        </div>

        {/* Fixed input container at bottom in response mode */}
        {response && (
          <div className={`response-mode-query ${isTransitioning ? 'entering' : ''}`}>
            <div className="query-input-container">
              <textarea
                className="query-input"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyPress}
                placeholder="Ask a follow-up question..."
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
        )}

        {/* Popups remain unchanged */}
        {activePopup && (
          <div className="popup-overlay" onClick={() => setActivePopup(null)}>
            <div className="popup-container" onClick={(e) => e.stopPropagation()}>
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
              {activePopup === 'recent' && (
                <div className="popup-content recent-highlights">
                  <div className="recent-highlights-header">
                    <h2>Recent Highlights</h2>
                    <span>{recentHighlights.length} of {queryHistory.length} items</span>
                  </div>
                  <div className="highlights-list-container">
                    {recentHighlights.length > 0 ? (
                      <ul className="highlights-list">
                        {recentHighlights.map((item) => (
                          <li key={item.id} className="highlight-item">
                            <div className="highlight-query">{item.query}</div>
                            {item.category && (
                              <div className="highlight-category">{item.category}</div>
                            )}
                            <div className="highlight-response">
                              {item.response.length > 200
                                ? `${item.response.substring(0, 200)}...`
                                : item.response}
                            </div>
                            <div className="highlight-actions">
                              <button
                                className="highlight-btn"
                                title="Use this query again"
                                onClick={() => {
                                  setQuery(item.query);
                                  setActivePopup(null);
                                }}
                              >
                                Reuse Query
                              </button>
                            </div>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <div className="no-highlights">
                        <div className="no-highlights-icon">📋</div>
                        <p>No recent queries found</p>
                      </div>
                    )}
                  </div>
                  <div className="popup-actions recent-highlights-actions">
                    <Link to="/history" className="view-all-link" onClick={() => setActivePopup(null)}>
                      <span>View All Queries</span>
                      <span>→</span>
                    </Link>
                    <button onClick={() => setActivePopup(null)} className="secondary-btn">
                      Close
                    </button>
                  </div>
                </div>
              )}
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
      </div>
    </div>
  );
};

export default Dashboard;