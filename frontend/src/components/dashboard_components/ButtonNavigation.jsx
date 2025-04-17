import React from "react";

const ButtonNavigation = ({ togglePopup }) => {
  return (
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
  );
};

export default ButtonNavigation;