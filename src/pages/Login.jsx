import React, { useEffect, useState } from 'react';
import { Card } from 'react-bootstrap';
import { Link, useLocation, useNavigate } from 'react-router-dom';

const Login = (props) => {
    const location = useLocation();
    const message = location.state?.message;

    const [credentials, setCredentials] = useState({ email: "", password: "" });
    const [showPassword, setShowPassword] = useState(false);
    let navigate = useNavigate();

    useEffect(() => {
        const token = sessionStorage.getItem('token');
        if (token) {
            props.showAlert && props.showAlert("Already Logged-in! (Logout to switch account)", 'info');
            navigate('/dashboard');
        }
        // Removed `props.showAlert` from dependencies to prevent infinite re-renders
    }, [navigate]);
    

    const handleSubmit = async (e) => {
        e.preventDefault();
        const response = await fetch("http://localhost:5000/api/auth/login", {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ email: credentials.email, password: credentials.password }),
        });
        const json = await response.json();
        console.log(json);
        if (json.success) {
            sessionStorage.setItem('token', json.authToken);
            props.showAlert("Logged-in Successfully", "success");
            navigate("/dashboard");
        } else {
            props.showAlert("Invalid credentials", "danger");
        }
    };

    const onChange = (e) => {
        setCredentials({ ...credentials, [e.target.name]: e.target.value });
    };

    const isFormValid = credentials.email && credentials.password;

    return (
        <div style={styles.container}>
            <div className="container d-flex justify-content-center align-items-center" style={styles.contentContainer}>
                <div style={styles.cardWrapper}>
                    <Card className="shadow-lg" style={styles.card}>
                        <div style={styles.titleContainer}>
                            <h1 style={styles.title}>Mr.Travis</h1>
                        </div>
                        <div style={styles.formContainer}>
                            <form onSubmit={handleSubmit}>
                                <div className="mb-4">
                                    <div className="input-group">
                                        <input
                                            type="email"
                                            className="form-control"
                                            style={styles.inputField}
                                            value={credentials.email}
                                            onChange={onChange}
                                            id="email"
                                            name="email"
                                            aria-describedby="emailHelp"
                                            placeholder="Email"
                                            required
                                        />
                                    </div>
                                    <div style={styles.helperText} id="emailHelp" className="form-text">
                                        We'll never share your email with anyone else.
                                    </div>
                                </div>
                                <div className="mb-4">
                                    <div className="input-group" style={styles.passwordInputGroup}>
                                        <input
                                            type={showPassword ? "text" : "password"}
                                            className="form-control"
                                            style={styles.inputField}
                                            value={credentials.password}
                                            onChange={onChange}
                                            name="password"
                                            id="password"
                                            placeholder="Password"
                                            required
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
                                <div className="mt-4">
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
                                        Login
                                    </button>
                                </div>
                            </form>
                        </div>
                        {message && (
                            <div style={styles.messageContainer}>
                                <p style={styles.errorMessage}>{message}</p>
                            </div>
                        )}
                    </Card>
                    <div style={styles.accountLinkContainer}>
                        <p style={styles.accountLink}>
                            Don't have an account?{' '}
                            <Link to="/signup" style={styles.link}>Sign up</Link>
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
        background: 'linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%)'
    },
    contentContainer: {
        minHeight: '80vh',
        width: '100%'
    },
    cardWrapper: {
        width: '100%',
        maxWidth: '420px',
        display: 'flex',
        flexDirection: 'column',
        gap: '1rem'
    },
    card: {
        borderRadius: '15px',
        overflow: 'hidden',
        border: 'none',
        boxShadow: '0 10px 25px rgba(0, 0, 0, 0.08)',
    },
    titleContainer: {
        background: 'linear-gradient(135deg, #009688 0%, #4DB6AC 100%)',
        padding: '1.5rem 0',
        textAlign: 'center',
        borderBottom: '4px solid #80CBC4'
    },
    title: {
        color: 'white',
        fontSize: '2.2rem',
        fontFamily: '"Bruno Ace SC", cursive',
        margin: 0,
        fontWeight: 'bold',
        textShadow: '1px 1px 3px rgba(0, 0, 0, 0.2)'
    },
    formContainer: {
        padding: '2rem',
        backgroundColor: 'transparent'
    },
    inputField: {
        height: '50px',
        borderRadius: '25px',
        padding: '0.75rem 1.25rem',
        fontSize: '1rem',
        border: '1px solid #ddd',
        boxShadow: 'inset 0 1px 3px rgba(0, 0, 0, 0.05)',
        transition: 'all 0.2s ease',
        backgroundColor: 'transparent'
    },
    passwordInputGroup: {
        position: 'relative'
    },
    passwordToggle: {
        position: 'absolute',
        right: '15px',
        top: '50%',
        transform: 'translateY(-50%)',
        cursor: 'pointer',
        zIndex: 10,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '0.5rem',
        color: '#555'
    },
    helperText: {
        fontSize: '0.85rem',
        color: '#6c757d',
        marginTop: '0.5rem',
        paddingLeft: '0.5rem'
    },
    button: {
        width: '100%',
        height: '50px',
        borderRadius: '25px',
        fontWeight: 'bold',
        fontSize: '1.1rem',
        color: 'white',
        border: 'none',
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
        transition: 'all 0.3s ease'
    },
    messageContainer: {
        padding: '0 2rem 1.5rem',
        textAlign: 'center'
    },
    errorMessage: {
        color: '#e53935',
        fontSize: '1rem',
        margin: 0
    },
    accountLinkContainer: {
        textAlign: 'center',
        marginTop: '1rem'
    },
    accountLink: {
        fontSize: '1rem',
        color: '#555'
    },
    link: {
        color: '#009688',
        fontWeight: 'bold',
        textDecoration: 'none',
        transition: 'color 0.2s ease'
    }
};

export default Login;