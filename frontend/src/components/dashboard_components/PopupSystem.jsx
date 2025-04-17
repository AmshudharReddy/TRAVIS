import React from "react";
import SpeakPopup from "./popups/SpeakPopup";
import RecentPopup from "./popups/RecentPopup";
import FontSizePopup from "./popups/FontSizePopup";
import SettingsPopup from "./popups/SettingsPopup";

const PopupSystem = ({
  activePopup,
  setActivePopup,
  recentHighlights,
  queryHistory,
  setQuery,
  fontSize,
  setFontSize,
  darkMode,
  setDarkMode,
  notificationsEnabled,
  setNotificationsEnabled,
  autoReadEnabled,
  setAutoReadEnabled
}) => {
  if (!activePopup) return null;

  return (
    <div className="popup-overlay" onClick={() => setActivePopup(null)}>
      <div className="popup-container" onClick={(e) => e.stopPropagation()}>
        {activePopup === 'speak' && (
          <SpeakPopup 
            onClose={() => setActivePopup(null)} 
            setQuery={setQuery}
          />
        )}
        {activePopup === 'recent' && (
          <RecentPopup 
            recentHighlights={recentHighlights} 
            queryHistory={queryHistory}
            setQuery={setQuery}
            onClose={() => setActivePopup(null)}
          />
        )}
        {activePopup === 'fontSize' && (
          <FontSizePopup 
            fontSize={fontSize}
            setFontSize={setFontSize}
            onClose={() => setActivePopup(null)}
          />
        )}
        {activePopup === 'settings' && (
          <SettingsPopup 
            darkMode={darkMode}
            setDarkMode={setDarkMode}
            notificationsEnabled={notificationsEnabled}
            setNotificationsEnabled={setNotificationsEnabled}
            autoReadEnabled={autoReadEnabled}
            setAutoReadEnabled={setAutoReadEnabled}
            onClose={() => setActivePopup(null)}
          />
        )}
      </div>
    </div>
  );
};

export default PopupSystem;