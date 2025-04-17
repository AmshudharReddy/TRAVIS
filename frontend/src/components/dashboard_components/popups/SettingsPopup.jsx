import React from "react";

const SettingsPopup = ({ 
  darkMode, 
  setDarkMode, 
  notificationsEnabled, 
  setNotificationsEnabled, 
  autoReadEnabled, 
  setAutoReadEnabled, 
  onClose 
}) => {
  return (
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
        <button onClick={onClose} className="secondary-btn">Close</button>
      </div>
    </div>
  );
};

export default SettingsPopup;