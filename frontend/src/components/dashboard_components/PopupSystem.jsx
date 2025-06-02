import React from "react";
// In PopupSystem component
import './PopupSystem.css';

// In VoiceInput component  
import './popups/SpeakPopup.css';

// In FontSizePopup component
import './popups/FontSizePopup.css';

// In RecentHighlights component
import './popups/RecentPopup.css';

// In SettingsPopup component
import './popups/SettingsPopup.css';

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
  transformerMode,
  setTransformerMode,
  autoReadEnabled,
  setAutoReadEnabled
}) => {
  const closePopup = () => setActivePopup(null);
  if (!activePopup) return null;

  return (
    <div className="popup-overlay" onClick={closePopup}>
      <div className="popup-container" onClick={(e) => e.stopPropagation()}>
        {activePopup === "speak" && (
          <SpeakPopup onClose={closePopup} setQuery={setQuery} />
        )}
        {activePopup === "recent" && (
          <RecentPopup
            recentHighlights={recentHighlights}
            queryHistory={queryHistory}
            setQuery={setQuery}
            onClose={closePopup}
          />
        )}
        {activePopup === "fontSize" && (
          <FontSizePopup
            fontSize={fontSize}
            setFontSize={setFontSize}
            onClose={closePopup}
          />
        )}
        {activePopup === "settings" && (
          <SettingsPopup
            darkMode={darkMode}
            setDarkMode={setDarkMode}
            transformerMode={transformerMode}
            setTransformerMode={setTransformerMode}
            autoReadEnabled={autoReadEnabled}
            setAutoReadEnabled={setAutoReadEnabled}
            onClose={closePopup}
          />
        )}
      </div>
    </div>
  );
};

export default PopupSystem;