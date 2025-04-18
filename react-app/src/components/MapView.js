import React, { useEffect, useRef } from "react";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import "leaflet-routing-machine"; // Import leaflet-routing-machine

// Fix for missing marker icons in Leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require("leaflet/dist/images/marker-icon-2x.png"),
  iconUrl: require("leaflet/dist/images/marker-icon.png"),
  shadowUrl: require("leaflet/dist/images/marker-shadow.png"),
});

const MapView = ({ userLocation, places }) => {
  const mapRef = useRef(null);
  const routingControlRef = useRef(null); // To hold the routing control

  useEffect(() => {
    if (userLocation && mapRef.current) {
      const map = mapRef.current;

      // Initializing the map
      map.setView([userLocation.lat, userLocation.lon], 14);

      // Adding user location marker
      const userMarker = L.marker([userLocation.lat, userLocation.lon]).addTo(map);
      userMarker.bindPopup("You are here").openPopup();

      // Remove previous route if exists
      if (routingControlRef.current) {
        routingControlRef.current.removeFrom(map); // Ensure that removeFrom works
      }
    }
  }, [userLocation]);

  const handleDirectionsClick = (hospital) => {
    if (userLocation) {
      const { lat, lon } = hospital;

      // Use Leaflet Routing Machine to add route
      const route = L.Routing.control({
        waypoints: [
          L.latLng(userLocation.lat, userLocation.lon),
          L.latLng(lat, lon),
        ],
        routeWhileDragging: true,
        createMarker: () => null, // Disable marker creation for the route
        showAlternatives: false, // Hide alternative routes
        lineOptions: {
          styles: [{ color: "#AA4A44", weight: 4 }],
        },
      }).addTo(mapRef.current);

      // Save the routing control for later removal
      routingControlRef.current = route;
    }
  };

  return (
    <MapContainer
      center={[userLocation.lat, userLocation.lon]}
      zoom={14}
      scrollWheelZoom={true}
      style={{ height: "400px", width: "100%" }}
      ref={mapRef}
    >
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      {/* User marker */}
      <Marker position={[userLocation.lat, userLocation.lon]}>
        <Popup>You are here</Popup>
      </Marker>

      {/* Hospital markers */}
      {places.map((place) => (
        <Marker key={place.id} position={[place.lat, place.lon]}>
          <Popup>
            <div>
              <strong>{place.name}</strong>
              <br />
              <button onClick={() => handleDirectionsClick(place)}>
                Get Directions
              </button>
            </div>
          </Popup>
        </Marker>
      ))}
    </MapContainer>
  );
};

export default MapView;
