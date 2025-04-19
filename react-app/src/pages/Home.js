import React from 'react';
import { Link } from 'react-router-dom';

const Home = () => {
    return (
        <div className="home-container">
            {/* Hero Section */}
            <section className="hero-section">
                <div className="hero-content">
                    <h1>AI-Powered Disease Prediction</h1>
                    <p>Get instant health insights and find nearby medical help</p>
                    <Link to="/predict" className="cta-button">Start Prediction</Link>
                </div>
            </section>

            {/* Features Section */}
            <section className="features-section">
                <h2>Our Services</h2>
                <div className="features-grid">
                    <div className="feature-card">
                        <div className="feature-icon">🔍</div>
                        <h3>Disease Prediction</h3>
                        <p>Advanced AI algorithms to analyze your symptoms</p>
                    </div>
                    <div className="feature-card">
                        <div className="feature-icon">👨‍⚕️</div>
                        <h3>Find Doctors</h3>
                        <p>Locate nearby healthcare professionals instantly</p>
                        <Link to="/nearby-doctors" className="feature-link">Find Doctors</Link>
                    </div>
                    <div className="feature-card">
                        <div className="feature-icon">📍</div>
                        <h3>Hospital Locator</h3>
                        <p>Discover medical facilities in your area</p>
                    </div>
                </div>
            </section>

            {/* How It Works Section */}
            <section className="how-it-works">
                <h2>How It Works</h2>
                <div className="steps-container">
                    <div className="step">
                        <div className="step-number">1</div>
                        <h3>Enter Symptoms</h3>
                        <p>Input your symptoms into our AI system</p>
                    </div>
                    <div className="step">
                        <div className="step-number">2</div>
                        <h3>Get Prediction</h3>
                        <p>Receive instant analysis and potential conditions</p>
                    </div>
                    <div className="step">
                        <div className="step-number">3</div>
                        <h3>Find Healthcare</h3>
                        <p>Connect with nearby medical professionals</p>
                    </div>
                </div>
            </section>
        </div>
    );
};

export default Home;
