import React from "react";

const SettingsPopup = ({
  darkMode,
  setDarkMode,
  transformerMode,
  setTransformerMode,
  autoReadEnabled,
  setAutoReadEnabled,
  onClose
}) => {
  const handleToggle = (key, currentValue, setter) => {
    if (typeof setter === "function") {
      const newValue = !currentValue;
      setter(newValue);
      localStorage.setItem(key, JSON.stringify(newValue));
    } else {
      console.error(`Invalid setter for ${key}`);
    }
  };

  return (
    <div className="popup-content">
      <h2>Settings</h2>
      <div className="settings-list">
        <div className="setting-item">
          <label>
            <span>Dark Mode</span>
            <button
              className={`toggle-switch ${darkMode ? "active" : ""}`}
              onClick={() => handleToggle("darkMode", darkMode, setDarkMode)}
            >
              {darkMode ? "ON" : "OFF"}
            </button>
          </label>
        </div>
        <div className="setting-item">
          <label>
            <span>Transformer Mode</span>
            <button
              className={`toggle-switch ${transformerMode ? "active" : ""}`}
              onClick={() =>
                handleToggle("transformerMode", transformerMode, setTransformerMode)
              }
            >
              {transformerMode ? "ON" : "OFF"}
            </button>
          </label>
        </div>
        <div className="setting-item">
          <label>
            <span>Auto-read Responses</span>
            <button
              className={`toggle-switch ${autoReadEnabled ? "active" : ""}`}
              onClick={() =>
                handleToggle("autoReadEnabled", autoReadEnabled, setAutoReadEnabled)
              }
            >
              {autoReadEnabled ? "ON" : "OFF"}
            </button>
          </label>
        </div>
      </div>
      <div className="popup-actions">
        <button onClick={onClose} className="secondary-btn">Close</button>
      </div>
    </div>
  );
};

export default SettingsPopup;