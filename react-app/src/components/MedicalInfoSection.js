import React, { useEffect, useState } from 'react';

const MedicalInfoSection = ({ disease }) => {
    const [data, setData] = useState({
        description: "Loading...",
        symptoms: [],
        treatment: [],
        medicines: [],
        rarity: "",
        criticality: ""
    });

    useEffect(() => {
        const fetchData = async () => {
            try {
                const res = await fetch("http://localhost:5000/generate_description", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ disease })
                });
                const json = await res.json();

                setData({
                    description: json.description,
                    symptoms: json.symptoms?.split(",").map(s => s.trim()) || [],
                    treatment: json.treatment?.split(",").map(s => s.trim()) || [],
                    medicines: json.medicines?.split(",").map(s => s.trim()) || [],
                    rarity: json.rarity,
                    criticality: json.criticality
                });
            } catch (err) {
                console.error(err);
                setData(prev => ({
                    ...prev,
                    description: "Failed to fetch disease details."
                }));
            }
        };

        if (disease) fetchData();
    }, [disease]);

    return (
        <div className="medical-info-section">
            <div className="info-card description">
                <h3>Details of primarily predicted Disease</h3>
                <p>{data.description}</p>
            </div>

            <div className="info-grid">
                <div className="info-card">
                    <h4>Common Symptoms</h4>
                    <div className="tag-container">
                        {data.symptoms.map((symptom, idx) => (
                            <span key={idx} className="info-tag">{symptom.replace('_', ' ')}</span>
                        ))}
                    </div>
                </div>

                <div className="info-card">
                    <h4>Treatment Options</h4>
                    <div className="tag-container">
                        {data.treatment.map((treatment, idx) => (
                            <span key={idx} className="info-tag">{treatment.replace(/[_()]/g, ' ')}</span>
                        ))}
                    </div>
                </div>

                <div className="info-card">
                    <h4>Recommended Medicines</h4>
                    <div className="tag-container">
                        {data.medicines.map((medicine, idx) => (
                            <span key={idx} className="info-tag">{medicine.replace('_', ' ')}</span>
                        ))}
                    </div>
                </div>

                <div className="info-card status">
                    <div className="status-item">
                        <h4>Rarity</h4>
                        <span className={`status-badge ${data.rarity}`}>{data.rarity}</span>
                    </div>
                    <div className="status-item">
                        <h4>Criticality</h4>
                        <span className={`status-badge ${data.criticality}`}>{data.criticality}</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default MedicalInfoSection;
