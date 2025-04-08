import React, { useEffect, useState } from 'react';
import { Link, useLocation, useNavigate } from "react-router-dom";
import { FaUserCircle, FaBars, FaTimes, FaSun, FaMoon } from "react-icons/fa";
import './Navbar.css';
import axios from 'axios';

const Navbar = ({ darkMode, setDarkMode }) => {
    const navigate = useNavigate();
    const location = useLocation();
    const [isMenuOpen, setIsMenuOpen] = useState(false);
    const [userName, setUserName] = useState("");

    useEffect(() => {
        const fetchUserDetails = async() => {
            try{
                const authToken = sessionStorage.getItem("auth-token");
                if(authToken){
                    const response = await axios.post('http://localhost:5000/api/auth/getuser', {}, {
                        headers: {
                            'auth-token': authToken
                        }
                    });
                    setUserName(response.data.name);
                }
            } catch(error){
                console.error("Error fetching the User Details: ", error);
            }
        };

        fetchUserDetails();
    }, []);

    const handleLogout = () => {
        sessionStorage.removeItem('auth-token');
        navigate('/login');
    };

    const toggleMenu = () => {
        setIsMenuOpen(!isMenuOpen);
    };

    const closeMenu = () => {
        setIsMenuOpen(false);
    };

    const toggleDarkMode = () => {
        const newDarkMode = !darkMode;
        setDarkMode(newDarkMode);
        localStorage.setItem('darkMode', newDarkMode);
        // Apply dark mode to the entire document body to ensure consistency
        if (newDarkMode) {
            document.body.classList.add('dark-mode');
        } else {
            document.body.classList.remove('dark-mode');
        }
    };

    // Apply dark mode to body on initial load
    useEffect(() => {
        if (darkMode) {
            document.body.classList.add('dark-mode');
        } else {
            document.body.classList.remove('dark-mode');
        }
    }, [darkMode]);

    return (
        <nav className={`navbar ${darkMode ? "dark-mode" : ""}`}>
            <div className="navbar-container">
                {/* Logo and Brand */}
                <Link to="/" className="logo-container">
                    <span className="logo-text">Mr.Travis</span>
                    <span className="tagline">AI Assistant</span>
                </Link>

                {/* Mobile Menu Toggle */}
                <button 
                    className="menu-toggle" 
                    onClick={toggleMenu}
                    aria-label="Toggle navigation menu"
                >
                    {isMenuOpen ? <FaTimes size={28} /> : <FaBars size={28} />}
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
                            {darkMode ? <FaSun size={24} /> : <FaMoon size={24} />}
                        </button>
                        
                        {sessionStorage.getItem('auth-token') ? (
                            <div className="user-section">
                                <Link to="/profile" className="profile-link">
                                    <FaUserCircle className="user-icon" />
                                    <span className="user-name">{userName ? `${userName}` : "Agent Athreya"}</span>
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