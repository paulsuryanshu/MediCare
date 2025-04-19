import React, { useState, useEffect } from 'react';
import { symptoms } from '../data/symptoms';
import { useNavigate } from 'react-router-dom';
import { usePrediction } from '../context/PredictionContext';

const Predict = () => {
    const [selectedSymptoms, setSelectedSymptoms] = useState([]);
    const [searchTerm, setSearchTerm] = useState('');
    const [customSymptom, setCustomSymptom] = useState('');
    const [filteredSymptoms, setFilteredSymptoms] = useState(symptoms);

    // Filter symptoms based on search term
    useEffect(() => {
        const filtered = symptoms.filter(symptom =>
            symptom.toLowerCase().includes(searchTerm.toLowerCase())
        );
        setFilteredSymptoms(filtered);
    }, [searchTerm]);

    // Handle symptom selection
    const toggleSymptom = (symptom) => {
        setSelectedSymptoms(prev => {
            if (prev.includes(symptom)) {
                return prev.filter(s => s !== symptom);
            } else {
                return [...prev, symptom];
            }
        });
    };

    // Handle custom symptom addition
    const [customSymptoms, setCustomSymptoms] = useState([]);
    const addCustomSymptom = (e) => {
        e.preventDefault();
        if (customSymptom.trim()) {
            setCustomSymptoms(prev => [...prev, customSymptom.trim()]);
            setCustomSymptom("");
        }
    };

    const { setPredictionData } = usePrediction(); // access context setter
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();

        const formData = new FormData();
        selectedSymptoms.forEach(symptom => {
            formData.append("symptoms", symptom);
        });
        formData.append("custom_symptoms", customSymptoms.join(","));

        try {
            const response = await fetch("https://medicare-u2v5.onrender.com", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
                console.error("Prediction error:", data.error);
                alert(data.error);
                return;
            }

            // ✅ Store result in context
            setPredictionData(data);

            // ✅ Navigate to results page
            navigate("/result");
        } catch (error) {
            console.error("Prediction error:", error);
            alert("Something went wrong with prediction.");
        }
    };

    return (
        <div className="predict-container">
            <div className="predict-header">
                <h1>Disease Prediction</h1>
                <p>Select your symptoms or add custom ones for analysis</p>
            </div>

            <div className="predict-content">
                {/* Search and Custom Symptom Input */}
                <div className="input-section">
                    <div className="search-box">
                        <input
                            type="text"
                            placeholder="Search symptoms..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            className="search-input"
                        />
                    </div>

                    <form onSubmit={addCustomSymptom} className="custom-symptom-form">
                        <input
                            type="text"
                            placeholder="Add a custom symptom..."
                            value={customSymptom}
                            onChange={(e) => setCustomSymptom(e.target.value)}
                            className="custom-input"
                        />
                        <button type="submit" className="add-button">Add</button>
                    </form>
                </div>

                {/* Selected Symptoms */}
                <div className="selected-symptoms">
                    <h3>Selected Symptoms ({selectedSymptoms.length})</h3>
                    <div className="selected-symptoms-list">
                        {selectedSymptoms.map(symptom => (
                            <div key={symptom} className="symptom-tag">
                                {symptom}
                                <button
                                    onClick={() => toggleSymptom(symptom)}
                                    className="remove-symptom"
                                >
                                    ×
                                </button>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Symptoms Selection */}
                <div className="symptoms-grid">
                    {filteredSymptoms.map(symptom => (
                        <button
                            key={symptom}
                            className={`symptom-button ${selectedSymptoms.includes(symptom) ? 'selected' : ''}`}
                            onClick={() => toggleSymptom(symptom)}
                        >
                            {symptom}
                        </button>
                    ))}
                </div>

                {/* Predict Button */}
                <div className="predict-action">
                    <button
                        onClick={handleSubmit}
                        disabled={selectedSymptoms.length === 0}
                        className="predict-button"
                    >
                        Get Prediction
                    </button>
                </div>
            </div>
        </div>
    );
};

export default Predict;
