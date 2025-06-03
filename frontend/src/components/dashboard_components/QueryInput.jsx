"use client"

import { useRef, useEffect, useState } from "react"
import { FaStop } from "react-icons/fa"
import './QueryInput.css';

const QueryInput = ({
  query,
  setQuery,
  onSubmit,
  placeholder = "Message TRAVIS...",
  isProcessing = false,
  transformerMode = true,
  setTransformerMode,
}) => {
  const textareaRef = useRef(null)
  const [isFocused, setIsFocused] = useState(false)

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = "auto"
      const newHeight = Math.min(Math.max(textarea.scrollHeight, 60), 200)
      textarea.style.height = `${newHeight}px`
    }
  }, [query])

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      if (query.trim() && !isProcessing) {
        onSubmit()
      }
    }
  }

  const handleSubmit = () => {
    if (query.trim() && !isProcessing) {
      onSubmit()
    }
  }

  const handleFocus = () => setIsFocused(true)
  const handleBlur = () => setIsFocused(false)

  // Handle mode selection with proper state management and validation
  const handleModeSelect = (mode) => {
    // Check if setTransformerMode function is available
    if (typeof setTransformerMode !== 'function') {
      console.warn('setTransformerMode is not a function. Mode switching is disabled.')
      return
    }

    if (mode === "database") {
      if (transformerMode !== false) {
        setTransformerMode(false)
        console.log("Switched to Database Mode (Transformer OFF)")
      }
    } else if (mode === "ai") {
      if (transformerMode !== true) {
        setTransformerMode(true)
        console.log("Switched to AI Mode (Transformer ON)")
      }
    }
  }

  // Determine which button is active based on transformerMode
  const isDatabaseActive = !transformerMode
  const isAIActive = transformerMode
  
  // Check if mode switching is available
  const isModeSwithingEnabled = typeof setTransformerMode === 'function'

  return (
    <div className={`modern-query-container ${isFocused ? "focused" : ""}`}>
      <div className="query-input-wrapper">
        {/* Main Input Area */}
        <textarea
          ref={textareaRef}
          className="modern-query-input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyPress}
          onFocus={handleFocus}
          onBlur={handleBlur}
          placeholder={placeholder}
          disabled={isProcessing}
          aria-label="Query input"
          rows={1}
        />

        {/* Mode Buttons - Bottom Left (Only show if mode switching is enabled) */}
        {isModeSwithingEnabled && (
          <div className="mode-buttons-bottom-left">
            {/* Database Mode Button */}
            <button
              type="button"
              className={`mode-toggle-btn database-mode ${isDatabaseActive ? "active" : ""}`}
              onClick={() => handleModeSelect("database")}
              title="Database Mode - Customer data and banking operations"
              disabled={isProcessing}
              aria-pressed={isDatabaseActive}
            >
              <div className="mode-icon">
                <svg width="16" height="16" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path
                    d="M10 2C6.5 2 3 3 3 4.5v11c0 1.5 3.5 2.5 7 2.5s7-1 7-2.5v-11C17 3 13.5 2 10 2z"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    fill="none"
                  />
                  <ellipse cx="10" cy="4.5" rx="7" ry="1.5" fill="currentColor" />
                  <path d="M3 8.5c0 1.5 3.5 2.5 7 2.5s7-1 7-2.5" stroke="currentColor" strokeWidth="1.5" fill="none" />
                  <path d="M3 12.5c0 1.5 3.5 2.5 7 2.5s7-1 7-2.5" stroke="currentColor" strokeWidth="1.5" fill="none" />
                </svg>
              </div>
              <span className="mode-label">Database</span>
            </button>

            {/* AI Mode Button */}
            <button
              type="button"
              className={`mode-toggle-btn ai-mode ${isAIActive ? "active" : ""}`}
              onClick={() => handleModeSelect("ai")}
              title="AI Mode - General queries and conversations"
              disabled={isProcessing}
              aria-pressed={isAIActive}
            >
              <div className="mode-icon">
                <svg width="16" height="16" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path
                    d="M2.656 17.344c-1.016-1.015-1.15-2.75-.313-4.925.325-.825.73-1.617 1.205-2.365L3.582 10l-.033-.054c-.5-.799-.91-1.596-1.206-2.365-.836-2.175-.703-3.91.313-4.926.56-.56 1.364-.86 2.335-.86 1.425 0 3.168.636 4.957 1.756l.053.034.053-.034c1.79-1.12 3.532-1.757 4.957-1.757.972 0 1.776.3 2.335.86 1.014 1.015 1.148 2.752.312 4.926a13.892 13.892 0 0 1-1.206 2.365l-.034.054.034.053c.5.8.91 1.596 1.205 2.365.837 2.175.704 3.911-.311 4.926-.56.56-1.364.861-2.335.861-1.425 0-3.168-.637-4.957-1.757L10 16.415l-.053.033c-1.79 1.12-3.532 1.757-4.957 1.757-.972 0-1.776-.3-2.335-.86z"
                    fill="currentColor"
                    stroke="currentColor"
                    strokeWidth=".1"
                  />
                  <path
                    d="M10.706 11.704A1.843 1.843 0 0 1 8.155 10a1.845 1.845 0 1 1 2.551 1.704z"
                    fill="currentColor"
                    stroke="currentColor"
                    strokeWidth=".2"
                  />
                </svg>
              </div>
              <span className="mode-label">AI-Mode</span>
            </button>
          </div>
        )}

        {/* Submit Button - Bottom Right */}
        <button
          onClick={handleSubmit}
          className={`modern-submit-btn ${query.trim() && !isProcessing ? "active" : ""} ${isProcessing ? "processing" : ""}`}
          title={isProcessing ? "Processing..." : "Send message"}
          disabled={!query.trim() || isProcessing}
          aria-label={isProcessing ? "Processing query" : "Send query"}
          style={{ 
            width: '42px', 
            height: '42px', 
            minWidth: '42px', 
            minHeight: '42px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          {isProcessing ? (
            <div className="processing-spinner">
              <FaStop size={20} />
            </div>
          ) : (
            <svg width="20" height="20" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path
                fillRule="evenodd"
                clipRule="evenodd"
                d="M8 16c-.595 0-1.077-.462-1.077-1.032V1.032C6.923.462 7.405 0 8 0s1.077.462 1.077 1.032v13.936C9.077 15.538 8.595 16 8 16z"
                fill="currentColor"
              />
              <path
                fillRule="evenodd"
                clipRule="evenodd"
                d="M1.315 8.44a1.002 1.002 0 0 1 0-1.46L7.238 1.302a1.11 1.11 0 0 1 1.523 0c.421.403.421 1.057 0 1.46L2.838 8.44a1.11 1.11 0 0 1-1.523 0z"
                fill="currentColor"
              />
              <path
                fillRule="evenodd"
                clipRule="evenodd"
                d="M14.685 8.44a1.11 1.11 0 0 1-1.523 0L7.238 2.762a1.002 1.002 0 0 1 0-1.46 1.11 1.11 0 0 1 1.523 0l5.924 5.678c.42.403.42 1.056 0 1.46z"
                fill="currentColor"
              />
            </svg>
          )}
        </button>
      </div>
    </div>
  )
}

export default QueryInput;