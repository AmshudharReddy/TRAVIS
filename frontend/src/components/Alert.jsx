import React, { useEffect, useState } from 'react';
import PropTypes from 'prop-types';

function Alert({ alert = null, duration = 3000 }) { // Default parameters here
    const [visible, setVisible] = useState(true);

    // Capitalize the alert type
    const capitalize = (word) => {
        if (!word) return '';
        if (word === 'danger') {
            word = 'error';
        }
        const lower = word.toLowerCase();
        return lower.charAt(0).toUpperCase() + lower.slice(1);
    };

    // Automatically, hide the alert after the specified duration
    useEffect(() => {
        if (alert && duration > 0) {
            const timer = setTimeout(() => {
                setVisible(false);
            }, duration);
            return () => clearTimeout(timer);
        }
    }, [alert, duration]);

    // Do not render if no alert is provided or if it's no longer visible
    if (!alert || !visible) return null;

    return (
        <div style={{ height: '60px', position: 'relative' }}>
            <div
                className={`alert alert-${alert.type} alert-dismissible fade show`}
                role="alert"
                style={{
                    textAlign: 'center',
                    margin: 'auto',
                    width: 'fit-content',
                    padding: '10px 20px',
                    borderRadius: '8px',
                    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
                    position: 'fixed',
                    top: '10vh',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    zIndex: 1050,
                }}
            >
                <strong>{capitalize(alert.type)}</strong>: {alert.msg}
            </div>
        </div>
    );
}

// PropTypes to validate the props
Alert.propTypes = {
    alert: PropTypes.shape({
        type: PropTypes.string.isRequired,
        msg: PropTypes.string.isRequired,
    }),
    duration: PropTypes.number, // Duration in milliseconds
};

export default Alert;