import React from "react";

const FontSizePopup = ({ fontSize, setFontSize, onClose }) => {
  return (
    <div className="popup-content">
      <h2>Font Size</h2>
      <div className="font-size-options">
        <button
          className={`font-option ${fontSize === 'small' ? 'active' : ''}`}
          onClick={() => setFontSize('small')}
        >
          Small
        </button>
        <button
          className={`font-option ${fontSize === 'medium' ? 'active' : ''}`}
          onClick={() => setFontSize('medium')}
        >
          Medium
        </button>
        <button
          className={`font-option ${fontSize === 'large' ? 'active' : ''}`}
          onClick={() => setFontSize('large')}
        >
          Large
        </button>
      </div>
      <div className="popup-actions">
        <button onClick={onClose} className="secondary-btn">Close</button>
      </div>
    </div>
  );
};

export default FontSizePopup;