// src/components/Results.js
import React from "react";

const Results = ({ places }) => {
  return (
    <div>
      <h3>Nearby Places</h3>
      <ul>
        {places.length === 0 && <li>No places found.</li>}
        {places.map((place) => (
          <li key={place.id}>
            📍 {place.name} <br />
            <small>
              Lat: {place.lat.toFixed(4)}, Lon: {place.lon.toFixed(4)}
            </small>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Results;
