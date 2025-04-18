import { createContext, useContext, useState } from "react";

const PredictionContext = createContext();

export const PredictionProvider = ({ children }) => {
    const [predictionData, setPredictionData] = useState(null);

    return (
        <PredictionContext.Provider value={{ predictionData, setPredictionData }}>
            {children}
        </PredictionContext.Provider>
    );
};

export const usePrediction = () => useContext(PredictionContext);
