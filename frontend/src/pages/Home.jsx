import React from 'react';
import { Container, Card, Button } from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';
import './Home.css';

const Home = ({ darkMode }) => {
  const navigate = useNavigate();

  const handleGetStarted = () => {
    const authToken = sessionStorage.getItem('token');
    if(!authToken){
      navigate('/login');
    } else{
      navigate('/dashboard');
    }
    
  };

  return (
    <div className={`home-container ${darkMode ? 'dark-mode' : ''}`}>
      <Container 
        className="d-flex align-items-center justify-content-center"
        style={{ height: '100vh', padding: 0, margin: 0, maxWidth: '100%' }}
      >
        <Card 
          className={`text-center p-5 shadow-lg ${darkMode ? 'dark-card' : ''}`}
          style={{ 
            maxWidth: '800px',
            backgroundColor: darkMode ? '#1e1e1e' : 'white'
          }}
        >
          <Card.Body>
            <Card.Title style={{ color: darkMode ? '#f8f9fa' : 'inherit' }}>Hello! I am</Card.Title>
            <Card.Header
              className={darkMode ? 'dark-header' : ''}
              style={{
                fontSize: '45px',
                fontFamily: 'monospace',
                fontWeight: 'bold',
                border: '40px',
                margin: '20px',
                backgroundColor: darkMode ? '#222' : 'whitesmoke',
                color: darkMode ? '#4ecca3' : 'teal'
              }}
            >
              TRAVIS, Agent Athreya
            </Card.Header>
            <Button
              variant={darkMode ? "outline-info" : "primary"}
              className={`mt-3 ${darkMode ? 'dark-button' : ''}`}
              style={{ 
                backgroundColor: darkMode ? 'transparent' : 'teal',
                borderColor: darkMode ? '#4ecca3' : 'teal',
                color: darkMode ? '#4ecca3' : 'white'
              }}
              onClick={handleGetStarted}
            >
              Get Started
            </Button>
          </Card.Body>
        </Card>
      </Container>
    </div>
  );
};

export default Home;