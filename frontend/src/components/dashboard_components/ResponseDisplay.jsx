import React, { useRef } from "react";
import { FaTimes, FaVolumeUp, FaLanguage } from "react-icons/fa";

const ResponseDisplay = ({ 
  lastQuery, 
  response, 
  responseCategory, 
  translatedResponse, 
  isTranslating, 
  onTranslate, 
  onClose, 
  speak 
}) => {
  const responseRef = useRef(null);
  const responseContentRef = useRef(null);

  return (
    <div className="response-mode" ref={responseRef}>
      <div className="response-header">
        <div className="query-display">
          <h3>{lastQuery}</h3>
          {responseCategory && <span className="response-category">{responseCategory}</span>}
        </div>
        <button onClick={onClose} className="close-response-btn" title="Close Response">
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
            onClick={onTranslate} 
            className="translate-btn" 
            title="Translate Response"
            disabled={isTranslating}
          >
            {isTranslating ? "Translating..." : <><FaLanguage />{'   '}Translate</>}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ResponseDisplay;