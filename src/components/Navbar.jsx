import React, { useState, useEffect } from 'react';
import { Link, useLocation, useNavigate } from "react-router-dom";
import { FaUserCircle, FaBars, FaTimes, FaSun, FaMoon } from "react-icons/fa";
import './Navbar.css';

const Navbar = ({ darkMode, setDarkMode }) => {
    const navigate = useNavigate();
    const location = useLocation();
    const [isMenuOpen, setIsMenuOpen] = useState(false);

    const handleLogout = () => {
        sessionStorage.removeItem('token');
        navigate('/login');
    };

    const toggleMenu = () => {
        setIsMenuOpen(!isMenuOpen);
    };

    const closeMenu = () => {
        setIsMenuOpen(false);
    };

    const toggleDarkMode = () => {
        setDarkMode(!darkMode);
        localStorage.setItem('darkMode', !darkMode);
    };

    return (
        <nav className={`navbar ${darkMode ? "dark-mode" : ""}`}>
            <div className="navbar-container">
                {/* Logo and Brand */}
                <Link to="/" className="logo-container">
                    <span className="logo-text">TRAVIS</span>
                    <span className="tagline">AI Assistant</span>
                </Link>

                {/* Mobile Menu Toggle */}
                <button 
                    className="menu-toggle" 
                    onClick={toggleMenu}
                    aria-label="Toggle navigation menu"
                >
                    {isMenuOpen ? <FaTimes size={24} /> : <FaBars size={24} />}
                </button>

                {/* Navigation Links and User Actions */}
                <div className={`nav-container ${isMenuOpen ? 'active' : ''}`}>
                    {/* Navigation Links */}
                    <ul className="nav-links">
                        <li className="nav-item">
                            <Link
                                to="/"
                                className={`nav-link ${location.pathname === "/" ? "active" : ""}`}
                                onClick={closeMenu}
                            >
                                Home
                            </Link>
                        </li>
                        <li className="nav-item">
                            <Link
                                to="/dashboard"
                                className={`nav-link ${location.pathname === "/dashboard" ? "active" : ""}`}
                                onClick={closeMenu}
                            >
                                Dashboard
                            </Link>
                        </li>
                        <li className="nav-item">
                            <Link
                                to="/about"
                                className={`nav-link ${location.pathname === "/about" ? "active" : ""}`}
                                onClick={closeMenu}
                            >
                                About
                            </Link>
                        </li>
                    </ul>

                    {/* Auth Section */}
                    <div className="auth-section">
                        {/* Dark Mode Toggle */}
                        <button className="theme-toggle" onClick={toggleDarkMode} aria-label="Toggle dark mode">
                            {darkMode ? <FaSun size={20} /> : <FaMoon size={20} />}
                        </button>
                        
                        {sessionStorage.getItem('token') ? (
                            <div className="user-section">
                                <Link to="/profile" className="profile-link">
                                    <FaUserCircle size={32} className="user-icon" />
                                    <span className="user-name">My Profile</span>
                                </Link>
                                <button
                                    onClick={handleLogout}
                                    className="logout-button"
                                >
                                    Logout
                                </button>
                            </div>
                        ) : (
                            <div className="auth-buttons">
                                <Link
                                    to="/login"
                                    className="login-button"
                                    onClick={closeMenu}
                                >
                                    Login
                                </Link>
                                <Link
                                    to="/signup"
                                    className="signup-button"
                                    onClick={closeMenu}
                                >
                                    Sign Up
                                </Link>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </nav>
    );
};

export default Navbar;