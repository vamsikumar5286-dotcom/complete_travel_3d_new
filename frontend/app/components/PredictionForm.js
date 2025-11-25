"use client";
import { useState } from "react";

export default function PredictionForm() {
  // Form state
  const [start, setStart] = useState("");
  const [destination, setDestination] = useState("");
  const [country, setCountry] = useState("");
  const [transport, setTransport] = useState("car");
  const [trainClass, setTrainClass] = useState("");
  const [days, setDays] = useState("");
  const [accommodation, setAccommodation] = useState("hostel");
  const [distance, setDistance] = useState("");

  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult("");

    // Build request body including Country
    const requestBody = {
      Start_Location: start,
      Destination: destination,
      Country: country,
      Transport_Type: transport,
      Train_Class: transport === "train" ? trainClass : null,
      Duration_Days: Number(days),
      Accommodation_Type: accommodation,
      Distance_km: Number(distance),
    };

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) throw new Error("Server error");

      const data = await response.json();
      setResult(`
Transport Cost: ${data.Transport_Cost}
Accommodation Cost: ${data.Accommodation_Cost}
Food Cost: ${data.Food_Cost}
Activity Cost: ${data.Activity_Cost}
Total Expense: ${data.Total_Expense}
      `);
    } catch (err) {
      setResult("Error: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ 
      position: "absolute", 
      top: 20, 
      left: 20, 
      background: "rgba(0,0,0,0.7)", 
      padding: "20px", 
      borderRadius: "8px", 
      color: "white", 
      maxWidth: "300px" 
    }}>
      <form onSubmit={handleSubmit}>
        {/* Start Location */}
        <input
          type="text"
          value={start}
          onChange={(e) => setStart(e.target.value)}
          placeholder="Start Location"
          style={{ padding: "8px", marginBottom: "8px", width: "100%", color: "black", backgroundColor: "white", borderRadius: "4px", border: "1px solid #ccc" }}
          required
        />

        {/* Destination */}
        <input
          type="text"
          value={destination}
          onChange={(e) => setDestination(e.target.value)}
          placeholder="Destination"
          style={{ padding: "8px", marginBottom: "8px", width: "100%", color: "black", backgroundColor: "white", borderRadius: "4px", border: "1px solid #ccc" }}
          required
        />

        {/* Country */}
        <input
          type="text"
          value={country}
          onChange={(e) => setCountry(e.target.value)}
          placeholder="Country"
          style={{ padding: "8px", marginBottom: "8px", width: "100%", color: "black", backgroundColor: "white", borderRadius: "4px", border: "1px solid #ccc" }}
          required
        />

        {/* Mode of Transportation */}
        <select
          value={transport}
          onChange={(e) => setTransport(e.target.value)}
          style={{ padding: "8px", marginBottom: "8px", width: "100%", borderRadius: "4px" }}
        >
          <option value="car">Car</option>
          <option value="bike">Bike</option>
          <option value="train">Train</option>
          <option value="flight">Flight</option>
          <option value="bus">Bus</option>
        </select>

        {/* Conditional: Train Class */}
        {transport === "train" && (
          <input
            type="text"
            value={trainClass}
            onChange={(e) => setTrainClass(e.target.value)}
            placeholder="Train Class (AC/Sleeper)"
            style={{ padding: "8px", marginBottom: "8px", width: "100%", color: "black", backgroundColor: "white", borderRadius: "4px", border: "1px solid #ccc" }}
            required
          />
        )}

        {/* Number of Days */}
        <input
          type="number"
          value={days}
          onChange={(e) => setDays(e.target.value)}
          placeholder="Number of Days"
          style={{ padding: "8px", marginBottom: "8px", width: "100%", color: "black", backgroundColor: "white", borderRadius: "4px", border: "1px solid #ccc" }}
          required
        />

        {/* Accommodation Type */}
        <select
          value={accommodation}
          onChange={(e) => setAccommodation(e.target.value)}
          style={{ padding: "8px", marginBottom: "8px", width: "100%", borderRadius: "4px" }}
        >
          <option value="hostel">Hostel</option>
          <option value="airbnb">Airbnb</option>
          <option value="hotel3">Hotel 3*</option>
          <option value="hotel4">Hotel 4*</option>
          <option value="hotel5">Hotel 5*</option>
        </select>

        {/* Distance */}
        <input
          type="number"
          value={distance}
          onChange={(e) => setDistance(e.target.value)}
          placeholder="Distance (km)"
          style={{ padding: "8px", marginBottom: "8px", width: "100%", color: "black", backgroundColor: "white", borderRadius: "4px", border: "1px solid #ccc" }}
          required
        />

        {/* Submit Button */}
        <button
          type="submit"
          style={{ padding: "10px 15px", width: "100%", backgroundColor: "#008cff", color: "white", border: "none", borderRadius: "4px", cursor: "pointer" }}
        >
          Estimate
        </button>
      </form>

      {/* Loading & Result */}
      {loading && <p>Calculating...</p>}
      {result && <pre style={{ marginTop: "10px" }}>{result}</pre>}
    </div>
  );
}