import React, { useState, useEffect } from "react";
import "./Dashboard.css";
import WelcomeMessage from "../../components/dashboard_components/WelcomeMessage";
import QueryInput from "../../components/dashboard_components/QueryInput";
import ResponseDisplay from "../../components/dashboard_components/ResponseDisplay";
import ButtonNavigation from "../../components/dashboard_components/ButtonNavigation";
import PopupSystem from "../../components/dashboard_components/PopupSystem";
import Footer from "../../components/dashboard_components/Footer";
import { containsSensitiveInfo } from "../../utils/securityUtils";

// API base
const API_BASE_URL = "http://localhost:5000";

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
  const [transformerMode, setTransformerMode] = useState(() => {
    // Try to get saved value from localStorage
    const saved = localStorage.getItem("transformerMode");
    return saved !== null ? JSON.parse(saved) : true; // DEFAULT: true
  });

  // Save to localStorage whenever transformerMode changes
  useEffect(() => {
    localStorage.setItem("transformerMode", JSON.stringify(transformerMode));
  }, [transformerMode]);

  // Clear account number on page refresh/reload
  useEffect(() => {
    const handleBeforeUnload = () => {
      localStorage.removeItem("savedAccountNumber");
    };

    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
    };
  }, []);

  // Reset state when transformer mode changes
  useEffect(() => {
    // Clear any pending queries and reset state when mode changes
    setQuery("");
    setResponse(null);
    setTranslatedResponse(null);
    setResponseCategory(null);
    setLastQuery("");
    localStorage.removeItem("savedAccountNumber");
  }, [transformerMode]);

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

  const trackQuery = (type, queryText, category) => {
    console.log(`Tracking ${type} query: ${queryText} (${category})`);
    fetchQueryHistory();
  };

  const logError = (errorType, errorMessage) => {
    console.error(`[${errorType}]`, errorMessage);
  };


const handleQuerySubmit = async (e) => {
  if (e) e.preventDefault();
  if (!query.trim()) return;

  const currentQuery = query.trim();
  const authToken = sessionStorage.getItem("auth-token");
  if (!authToken) {
    showAlert("You need to be logged in to use this feature.");
    return;
  }

  console.log("Submitting query:", currentQuery, "Mode:", transformerMode ? "transformer" : "customer");

  setLastQuery(currentQuery);
  setQuery("");
  setTranslatedResponse(null);

  const isSecureQuery = containsSensitiveInfo(currentQuery);
  const BASE_ROUTE = transformerMode ? "query" : "customers";

  try {
    if (!transformerMode && isSecureQuery) {
      // Handle secure queries
      const accountNumber = await promptForAccountNumber();
      if (!accountNumber) {
        return;
      }

      console.log("Making secure query request to:", `${API_BASE_URL}/api/${BASE_ROUTE}/secureQuery`);
      const response = await fetch(`${API_BASE_URL}/api/${BASE_ROUTE}/secureQuery`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "auth-token": authToken,
        },
        body: JSON.stringify({ query: currentQuery, accountNumber }),
      });

      const data = await response.json();
      if (response.ok) {
        setResponse(data.response || "No response received");
        setResponseCategory(data.category || "Secure");
        if (autoReadEnabled) speak(data.response);
        trackQuery("secure", currentQuery, data.category);
      } else {
        setResponse(data.error || "An error occurred while processing the query");
        setResponseCategory("Error");
        logError("Secure query error", data.error);
      }
    } else {
      // Handle transformer mode or non-secure customer queries
      console.log("Making general query request to:", `${API_BASE_URL}/api/${BASE_ROUTE}/`);
      // console.log("Making category prediction request to:", `http://127.0.0.1:5001/api/classify`);

      // Perform both requests concurrently
      const [queryResponse, catResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/api/${BASE_ROUTE}/`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "auth-token": authToken,
          },
          body: JSON.stringify({ query: currentQuery }),
        }).then(res => res.json().then(data => ({ ok: res.ok, data }))),
        fetch(`${API_BASE_URL}/api/${BASE_ROUTE}/category`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "auth-token": authToken,
          },
          body: JSON.stringify({ query: currentQuery }),
        }).then(res => res.json().then(data => ({ ok: res.ok, data }))),
      ]);

      // Handle query response
      if (queryResponse.ok) {
        setResponse(queryResponse.data.response || "No response received");
      } else {
        setResponse(queryResponse.data.error || "An error occurred while processing the query");
        setResponseCategory("Error");
        logError("General query error", queryResponse.data.error);
        return; // Exit early if query response fails
      }

      // Handle category response
      if (catResponse.ok) {
        setResponseCategory(catResponse.data.category || "General");
      } else {
        setResponseCategory("General"); // Fallback category
        logError("Category prediction error", catResponse.data.error || "Unknown error");
      }

      // Additional actions if query response was successful
      if (autoReadEnabled) speak(queryResponse.data.response);
      trackQuery("general", currentQuery, catResponse.ok ? catResponse.data.category : "General");
    }
  } catch (error) {
    console.error("Request failed:", error);
    setResponse("Network error. Please try again later.");
    setResponseCategory("Error");
    logError("Network error", error.message);
  }
};

  const promptForAccountNumber = () => {
    return new Promise((resolve) => {
      const storedAccountNumber = localStorage.getItem("savedAccountNumber");

      // For follow-up query, use the stored account number
      if (storedAccountNumber && response && /^[A-Z]{2}\d{10}$/.test(storedAccountNumber)) {
        resolve(storedAccountNumber);
        return;
      }

      // Prompt for new account number
      const accountInput = prompt("Please enter your 12-character account number (e.g., IN1234567890):");
      if (accountInput === null) {
        showAlert("Authentication canceled. Unable to process secure query.");
        resolve(null);
        return;
      }

      if (!/^[A-Z]{2}\d{10}$/.test(accountInput)) {
        showAlert("Invalid account number format.");
        resolve(null);
        return;
      }

      // Save for session use
      localStorage.setItem("savedAccountNumber", accountInput);
      resolve(accountInput);
    });
  };

  const handleTranslate = async () => {
    if (!response) return;

    setIsTranslating(true);
    const authToken = sessionStorage.getItem("auth-token");
    if (!authToken) {
      setIsTranslating(false);
      return;
    }

    try {
      const translatedResponse = await fetch(`${API_BASE_URL}/api/query/translate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "auth-token": authToken,
        },
        body: JSON.stringify({ response }),
      });

      const data = await translatedResponse.json();
      if (translatedResponse.ok) {
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

    localStorage.removeItem("savedAccountNumber"); // clear account number
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
    setActivePopup(activePopup === popupName ? null : popupName);
  };

  const fetchQueryHistory = () => {
    const authToken = sessionStorage.getItem("auth-token");
    if (!authToken) return;

    fetch(`${API_BASE_URL}/api/query/history?limit=3&sort=-createdAt`, {
      headers: { "auth-token": authToken }
    })
      .then(res => res.json())
      .then(data => {
        setQueryHistory(data);
        setRecentHighlights(data.slice(0, 3).map((entry, index) => ({
          id: index + 1,
          query: entry.query,
          response: entry.response,
          category: entry.category || "General"
        })));
      })
      .catch(err => console.error("Error fetching history:", err));
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
                transformerMode={transformerMode}
                setTransformerMode={setTransformerMode}
              />
            )}
            {!response && <ButtonNavigation togglePopup={togglePopup} />}
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
              // isProcessing={isProcessing}
              transformerMode={transformerMode}
              setTransformerMode={setTransformerMode}
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
          transformerMode={transformerMode}
          setTransformerMode={setTransformerMode}
        />
      </div>
    </div>
  );
};

export default Dashboard;