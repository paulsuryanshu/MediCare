import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import Home from "./pages/Home";
import NearbyDoctors from "./pages/NearbyDoctors";
import Predict from "./pages/Predict";
import Result from "./pages/Result"; // <-- Import the result page

import { BrowserRouter, Route, Routes } from "react-router-dom";
import { PredictionProvider } from "./context/PredictionContext";

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <BrowserRouter>
    <PredictionProvider>
      <Routes>
        <Route path="/" element={<App />}>
          <Route index element={<Home />} />
          <Route path="predict" element={<Predict />} />
          <Route path="result" element={<Result />} /> {/* <-- Add this */}
          <Route path="nearby-doctors" element={<NearbyDoctors />} />
        </Route>
      </Routes>
    </PredictionProvider>
  </BrowserRouter>
);
