import React, { useEffect, useState } from 'react'
import MapView from '../components/MapView';
import Results from '../components/Results';
import { getNearbyPlaces } from '../services/overpass';

const NearbyDoctors = () => {
    const [userLocation, setUserLocation] = useState(null);
    const [places, setPlaces] = useState([]);
    const [loading, setLoading] = useState(true);
    const [locationError, setLocationError] = useState(false); // For tracking location errors
    const [errorMessage, setErrorMessage] = useState(""); // To store any error messages

    const fetchNearbyPlaces = async (lat, lon) => {
        try {
            const nearby = await getNearbyPlaces(lat, lon, "hospital");
            if (nearby && nearby.length > 0) {
                setPlaces(nearby);
            } else {
                setErrorMessage("No nearby hospitals found.");
            }
        } catch (error) {
            console.error("Error fetching nearby places:", error);
            setErrorMessage("Error fetching nearby hospitals. Please try again later.");
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        navigator.geolocation.getCurrentPosition(
            (position) => {
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                setUserLocation({ lat, lon });

                // Fetch nearby places based on user location
                fetchNearbyPlaces(lat, lon);
            },
            (error) => {
                console.error("Geolocation error:", error);
                setLocationError(true); // Set location error state
                setLoading(false);

                // Fallback to a default location (New Delhi as an example)
                const fallbackLocation = { lat: 28.6139, lon: 77.2090 }; // New Delhi coordinates
                setUserLocation(fallbackLocation);

                // Fetch nearby places based on fallback location
                fetchNearbyPlaces(fallbackLocation.lat, fallbackLocation.lon);
            }
        );
    }, []);

    return (
        <div style={{ padding: "20px" }}>
            <h2 style={{ marginBottom: "10px" }}>Find Nearby Doctors</h2>

            {loading && !locationError && <p>Loading...</p>}
            {locationError && !loading && (
                <p style={{ color: "red" }}>
                    Could not fetch your location. Using fallback location (New Delhi).
                </p>
            )}

            {errorMessage && <p style={{ color: "red" }}>{errorMessage}</p>}

            {!loading && userLocation && (
                <>
                    <MapView userLocation={userLocation} places={places} />
                    <Results places={places} />
                </>
            )}
        </div>
    );
}

export default NearbyDoctors