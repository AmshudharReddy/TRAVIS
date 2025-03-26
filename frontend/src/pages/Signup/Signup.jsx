import React, { useCallback, useEffect, useState } from 'react';
import { Card } from 'react-bootstrap';
import { Link, useNavigate } from 'react-router-dom';

const Signup = (props) => {
    const [credentials, setCredentials] = useState({ name: "", email: "", password: "", cpassword: "" });
    const [showPassword, setShowPassword] = useState(false);
    const [passwordMatch, setPasswordMatch] = useState(true);
    let navigate = useNavigate();

    const { showAlert } = props;

    const memoizedShowAlert = useCallback(() => {
        showAlert?.("This user Exists! (Please try with another)", "info");
    }, [showAlert]);

    // Check login status only once when the component mounts
    useEffect(() => {
        const token = sessionStorage.getItem("auth-token");
        if (token) {
            setTimeout(() => {
                memoizedShowAlert();
            }, 100);
            navigate("/dashboard");
        }
    }, []); // Empty dependency array ensures it runs only once

    //  Password confirmation logic
    useEffect(() => {
        setPasswordMatch(credentials.password === credentials.cpassword);
    }, [credentials.password, credentials.cpassword]);

    const handleKeyDown = (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            const form = e.target.form;
            const index = Array.from(form).indexOf(e.target);
            if (form[index + 1]) {
                form[index + 1].focus();
            } else {
                form.requestSubmit();
            }
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        const { name, email, password } = credentials;
        const response = await fetch("http://localhost:5000/api/auth/signup", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ name, email, password }),
        });
        const json = await response.json();
        console.log(json);

        if (json.success) {
            sessionStorage.setItem('auth-token', json.authToken);
            showAlert("Account created Successfully", "success");
            navigate("/dashboard");
        } else {
            showAlert("Invalid details", "danger");
        }
    };

    const onChange = (e) => {
        setCredentials({ ...credentials, [e.target.name]: e.target.value });
    };

    const isFormValid = credentials.name &&
        credentials.email &&
        credentials.password &&
        credentials.cpassword &&
        passwordMatch &&
        credentials.password.length >= 5;

    return (
        <div style={styles.container}>
            {/* Back Button */}
            <Link to="/" style={styles.backButton}>
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M19 12H5M12 19l-7-7 7-7" />
                </svg>
                <span>Back</span>
            </Link>

            <div className="d-flex justify-content-center align-items-center" style={styles.contentContainer}>
                <div style={styles.cardWrapper}>
                    <Card style={styles.card}>
                        <div style={styles.titleContainer}>
                            <h1 style={styles.title}>Mr.Travis</h1>
                        </div>
                        <div style={styles.formContainer}>
                            <form onSubmit={handleSubmit}>
                                <div className="mb-3">
                                    <input
                                        type="text"
                                        className="form-control"
                                        style={styles.inputField}
                                        value={credentials.name}
                                        onChange={onChange}
                                        onKeyDown={handleKeyDown}
                                        id="name"
                                        name="name"
                                        placeholder="Full Name"
                                        required
                                    />
                                </div>
                                <div className="mb-3">
                                    <input
                                        type="email"
                                        className="form-control"
                                        style={styles.inputField}
                                        value={credentials.email}
                                        onChange={onChange}
                                        onKeyDown={handleKeyDown}
                                        id="email"
                                        name="email"
                                        placeholder="Email"
                                        required
                                    />
                                    <div style={styles.helperText} id="emailHelp" className="form-text">
                                        We'll never share your email with anyone else.
                                    </div>
                                </div>
                                <div className="mb-3">
                                    <div className="input-group" style={styles.passwordInputGroup}>
                                        <input
                                            type={showPassword ? "text" : "password"}
                                            className="form-control"
                                            style={styles.inputField}
                                            value={credentials.password}
                                            onChange={onChange}
                                            onKeyDown={handleKeyDown}
                                            name="password"
                                            id="password"
                                            placeholder="Password (min 5 characters)"
                                            required
                                            minLength="5"
                                        />
                                        <span
                                            style={styles.passwordToggle}
                                            onClick={() => setShowPassword(!showPassword)}
                                            aria-label={showPassword ? "Hide password" : "Show password"}
                                        >
                                            {showPassword ? (
                                                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                                                    <circle cx="12" cy="12" r="3"></circle>
                                                </svg>
                                            ) : (
                                                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                                    <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path>
                                                    <line x1="1" y1="1" x2="23" y2="23"></line>
                                                </svg>
                                            )}
                                        </span>
                                    </div>
                                </div>
                                <div className="mb-3">
                                    <input
                                        type="password"
                                        className={`form-control ${!passwordMatch && credentials.cpassword ? 'is-invalid' : ''}`}
                                        style={{
                                            ...styles.inputField,
                                            borderColor: !passwordMatch && credentials.cpassword ? '#dc3545' : '#ddd'
                                        }}
                                        value={credentials.cpassword}
                                        onChange={onChange}
                                        onKeyDown={handleKeyDown}
                                        name="cpassword"
                                        id="cpassword"
                                        placeholder="Confirm Password"
                                        required
                                    />
                                    {!passwordMatch && credentials.cpassword && (
                                        <div className="invalid-feedback" style={styles.errorFeedback}>
                                            Passwords do not match
                                        </div>
                                    )}
                                </div>
                                <div className="mt-3">
                                    <button
                                        type="submit"
                                        className="btn"
                                        style={{
                                            ...styles.button,
                                            backgroundColor: isFormValid ? '#009688' : '#80CBC4',
                                            opacity: isFormValid ? 1 : 0.7
                                        }}
                                        disabled={!isFormValid}
                                    >
                                        Create Account
                                    </button>
                                </div>
                            </form>
                        </div>
                    </Card>
                    <div style={styles.accountLinkContainer}>
                        <p style={styles.accountLink}>
                            Already have an account?{' '}
                            <Link to="/login" style={styles.link}>Log in</Link>
                        </p>
                        <p style={styles.accountLink}>
                            Go back to{' '}
                            <Link to="/" style={styles.link}>Home</Link>
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
};

const styles = {
    container: {
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '100vh',
        fontFamily: '"Kumbh Sans", sans-serif',
        background: 'none', // Removed background gradient for better dark mode compatibility
        position: 'relative'
    },
    contentContainer: {
        minHeight: '80vh',
        width: '100%'
    },
    cardWrapper: {
        width: '100%',
        maxWidth: '400px',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.5rem'
    },
    card: {
        borderRadius: '12px',
        overflow: 'hidden',
        border: 'none',
        boxShadow: '0 5px 15px rgba(0, 0, 0, 0.1)',
        background: 'inherit' // Use inherit to respect the theme
    },
    titleContainer: {
        background: 'linear-gradient(135deg, #009688 0%, #4DB6AC 100%)',
        padding: '1.25rem 0',
        textAlign: 'center',
        borderBottom: '3px solid #80CBC4'
    },
    title: {
        color: 'white',
        fontSize: '2rem',
        fontFamily: '"Bruno Ace SC", cursive',
        margin: 0,
        fontWeight: 'bold',
        textShadow: '1px 1px 2px rgba(0, 0, 0, 0.2)'
    },
    formContainer: {
        padding: '1.5rem',
        backgroundColor: 'transparent'
    },
    inputField: {
        height: '45px',
        borderRadius: '22px',
        padding: '0.5rem 1rem',
        fontSize: '1rem',
        border: '1px solid #ddd',
        boxShadow: 'none',
        transition: 'all 0.2s ease',
        backgroundColor: 'transparent'
    },
    passwordInputGroup: {
        position: 'relative'
    },
    passwordToggle: {
        position: 'absolute',
        right: '12px',
        top: '50%',
        transform: 'translateY(-50%)',
        cursor: 'pointer',
        zIndex: 10,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '0.25rem',
        color: '#555'
    },
    helperText: {
        fontSize: '0.8rem',
        color: '#6c757d',
        marginTop: '0.25rem',
        paddingLeft: '0.5rem'
    },
    errorFeedback: {
        fontSize: '0.8rem',
        paddingLeft: '0.5rem'
    },
    button: {
        width: '100%',
        height: '45px',
        borderRadius: '22px',
        fontWeight: 'bold',
        fontSize: '1rem',
        color: 'white',
        border: 'none',
        boxShadow: '0 3px 5px rgba(0, 0, 0, 0.1)',
        transition: 'all 0.3s ease'
    },
    accountLinkContainer: {
        textAlign: 'center',
        marginTop: '0.5rem'
    },
    accountLink: {
        fontSize: '0.9rem',
        color: 'inherit', // Use inherit to respect the theme
        marginBottom: '0.25rem'
    },
    link: {
        color: '#009688',
        fontWeight: 'bold',
        textDecoration: 'none',
        transition: 'color 0.2s ease'
    },
    backButton: {
        position: 'absolute',
        top: '15px',
        left: '15px',
        display: 'flex',
        alignItems: 'center',
        gap: '5px',
        padding: '6px 10px',
        backgroundColor: '#009688',
        color: 'white',
        borderRadius: '20px',
        textDecoration: 'none',
        boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
        zIndex: 10,
        transition: 'all 0.2s ease',
        fontWeight: '500',
        fontSize: '0.9rem'
    }
};

export default Signup;