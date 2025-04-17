import React from "react";
import { Link } from 'react-router-dom';

const RecentPopup = ({ recentHighlights, queryHistory, setQuery, onClose }) => {
  return (
    <div className="popup-content recent-highlights">
      <div className="recent-highlights-header">
        <h2>Recent Highlights</h2>
        <span>{recentHighlights.length} of {queryHistory.length} items</span>
      </div>
      <div className="highlights-list-container">
        {recentHighlights.length > 0 ? (
          <ul className="highlights-list">
            {recentHighlights.map((item) => (
              <li key={item.id} className="highlight-item">
                <div className="highlight-query">{item.query}</div>
                {item.category && (
                  <div className="highlight-category">{item.category}</div>
                )}
                <div className="highlight-response">
                  {item.response.length > 200
                    ? `${item.response.substring(0, 200)}...`
                    : item.response}
                </div>
                <div className="highlight-actions">
                  <button
                    className="highlight-btn"
                    title="Use this query again"
                    onClick={() => {
                      setQuery(item.query);
                      onClose();
                    }}
                  >
                    Reuse Query
                  </button>
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <div className="no-highlights">
            <div className="no-highlights-icon">📋</div>
            <p>No recent queries found</p>
          </div>
        )}
      </div>
      <div className="popup-actions recent-highlights-actions">
        <Link to="/history" className="view-all-link" onClick={onClose}>
          <span>View All Queries</span>
          <span>→</span>
        </Link>
        <button onClick={onClose} className="secondary-btn">
          Close
        </button>
      </div>
    </div>
  );
};

export default RecentPopup;