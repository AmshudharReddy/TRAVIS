import React from "react";
import './QueryInput.css';
import { FaPaperPlane } from "react-icons/fa";

const QueryInput = ({ query, setQuery, onSubmit, placeholder = "Ask me anything..." }) => {
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSubmit();
    }
  };

  return (
    <div className="query-input-container">
      <textarea
        className="query-input"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onKeyDown={handleKeyPress}
        placeholder={placeholder}
      />
      <button
        onClick={onSubmit}
        className="process-btn"
        title="Process Query"
        disabled={query.trim() === ""}
      >
        <FaPaperPlane />
      </button>
    </div>
  );
};

export default QueryInput;