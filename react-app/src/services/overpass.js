// src/services/overpass.js
import axios from "axios";

// Overpass QL query builder
export async function getNearbyPlaces(lat, lon, type = "doctors") {
  const radius = 3000; // 3 km

  const query = `
    [out:json];
    (
      node["amenity"="${type}"](around:${radius},${lat},${lon});
      way["amenity"="${type}"](around:${radius},${lat},${lon});
      relation["amenity"="${type}"](around:${radius},${lat},${lon});
    );
    out center;
  `;

  try {
    const response = await axios.post(
      "https://overpass-api.de/api/interpreter",
      new URLSearchParams({ data: query })
    );

    return response.data.elements.map((el) => ({
      id: el.id,
      lat: el.lat || el.center?.lat,
      lon: el.lon || el.center?.lon,
      name: el.tags?.name || "Unnamed",
    }));
  } catch (error) {
    console.error("Overpass API error:", error);
    return [];
  }
}
