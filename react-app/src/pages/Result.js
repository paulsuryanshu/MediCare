import React from 'react';
import { usePrediction } from "../context/PredictionContext";
import MedicalInfoSection from '../components/MedicalInfoSection';

const Result = () => {
    const { predictionData } = usePrediction();

    if (!predictionData) {
        return (
            <div className="result-empty-state">
                <h2>No prediction data found.</h2>
                <p>Please complete a prediction first.</p>
            </div>
        );
    }

    const { prediction, graph_image, symptoms_chart, symptoms_reported } = predictionData;

    return (
        <div className="result-container">
            <div className="result-header">
                <h2>Disease Prediction Result</h2>
                <div className="result-summary">
                    <p>Based on {symptoms_reported.length} reported symptoms</p>
                </div>
            </div>

            <div className="result-grid">
                <div className="result-section symptoms-section">
                    <h3>Reported Symptoms</h3>
                    <div className="symptoms-list">
                        {symptoms_reported.map((symptom, idx) => (
                            <div key={idx} className="symptom-item">
                                <span className="symptom-bullet">•</span>
                                {symptom}
                            </div>
                        ))}
                    </div>
                </div>

                <div className="result-section prediction-section">
                    <div className="primary-prediction">
                        <h3>Primary Prediction</h3>
                        <div className="prediction-card primary">
                            <div className="disease-name">
                                {prediction.primary_prediction.disease}
                            </div>
                            <div className="confidence-score">
                                <div className="confidence-bar"
                                    style={{ width: `${prediction.primary_prediction.confidence * 100}%` }}>
                                </div>
                                <span className="confidence-value">
                                    {(prediction.primary_prediction.confidence * 100).toFixed(2)}%
                                </span>
                            </div>
                        </div>
                    </div>

                    <div className="alternative-predictions">
                        <h4>Alternative Predictions</h4>
                        <div className="alternatives-list">
                            {prediction.alternative_predictions.map((alt, idx) => (
                                <div key={idx} className="prediction-card alternative">
                                    <div className="disease-name">{alt.disease}</div>
                                    <div className="confidence-score">
                                        <div className="confidence-bar"
                                            style={{ width: `${alt.confidence * 100}%` }}>
                                        </div>
                                        <span className="confidence-value">
                                            {(alt.confidence * 100).toFixed(2)}%
                                        </span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>

            {/* New Medical Information Section */}
            <MedicalInfoSection disease={prediction.primary_prediction.disease} />

            <div className="visualization-section">
                <div className="chart-container">
                    <h3>Prediction Analysis</h3>
                    <img
                        src={`data:image/png;base64,${graph_image}`}
                        alt="Prediction Graph"
                        className="result-chart"
                    />
                </div>
                <div className="chart-container">
                    <h3>Symptoms Analysis</h3>
                    <img
                        src={`data:image/png;base64,${symptoms_chart}`}
                        alt="Symptoms Chart"
                        className="result-chart"
                    />
                </div>
            </div>
        </div>
    );
};

export default Result;
