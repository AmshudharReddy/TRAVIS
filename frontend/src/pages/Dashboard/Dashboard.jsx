// Main Dashboard.jsx file
import React, { useState, useEffect } from "react";
import "./Dashboard.css";
import WelcomeMessage from "../../components/dashboard_components/WelcomeMessage";
import QueryInput from "../../components/dashboard_components/QueryInput";
import ResponseDisplay from "../../components/dashboard_components/ResponseDisplay";
import ButtonNavigation from "../../components/dashboard_components/ButtonNavigation";
import PopupSystem from "../../components/dashboard_components/PopupSystem";
import Footer from "../../components/dashboard_components/Footer";

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
  const [queryHistory, setQueryHistory] = useState([]);
  const [recentHighlights, setRecentHighlights] = useState([]);

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
    setTranslatedResponse(null);

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

  useEffect(() => {
    localStorage.setItem('fontSize', fontSize);
  }, [fontSize]);

  useEffect(() => {
    fetchQueryHistory();
  }, []);

  return (
    <div className={dashboardClasses}>
      <div className="dashboard-inner">
        <div className="conversation-container">
          <main className="main-content">
            {!response && <WelcomeMessage />}

            {response && (
              <ResponseDisplay
                lastQuery={lastQuery}
                response={response}
                responseCategory={responseCategory}
                translatedResponse={translatedResponse}
                isTranslating={isTranslating}
                onTranslate={handleTranslate}
                onClose={handleCloseResponse}
                speak={speak}
              />
            )}

            {!response && <div className="query-label"><p style={{ fontSize: "20px", textAlign: "center" }}>What's your query?</p></div>}

            {!response && (
              <QueryInput
                query={query}
                setQuery={setQuery}
                onSubmit={handleQuerySubmit}
              />
            )}

            {!response && (
              <ButtonNavigation
                togglePopup={togglePopup}
              />
            )}
            
            {!response && <Footer />}
            
            {response && <div className="response-bottom-spacer"></div>}
          </main>
        </div>

        {response && (
          <div className={`response-mode-query ${isTransitioning ? 'entering' : ''}`}>
            <QueryInput
              query={query}
              setQuery={setQuery}
              onSubmit={handleQuerySubmit}
              placeholder="Ask a follow-up question..."
            />
          </div>
        )}

        <PopupSystem
          activePopup={activePopup}
          setActivePopup={setActivePopup}
          recentHighlights={recentHighlights}
          queryHistory={queryHistory}
          setQuery={setQuery}
          fontSize={fontSize}
          setFontSize={setFontSize}
          darkMode={darkMode}
          setDarkMode={setDarkMode}
          notificationsEnabled={notificationsEnabled}
          setNotificationsEnabled={setNotificationsEnabled}
          autoReadEnabled={autoReadEnabled}
          setAutoReadEnabled={setAutoReadEnabled}
        />
      </div>
    </div>
  );
};

export default Dashboard;