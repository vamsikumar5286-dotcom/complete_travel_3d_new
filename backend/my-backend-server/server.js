import express from 'express';
import bodyParser from 'body-parser';
import cors from 'cors';

const app = express();
const port = 3001;

// Middleware setup
app.use(cors()); // Enable CORS for the frontend running on a different port/origin
app.use(bodyParser.json());

// --- Cost Configuration (Rule-Based Costs) ---

// Daily food costs based on user preference
const FOOD_COSTS = {
    'Low': 7,    // $7 per day
    'Medium': 11, // $11 per day
    'High': 25   // $25 per day
};

// Daily activity costs based on user preference
const ACTIVITY_COSTS = {
    'Low': 12,   // $12 per day
    'Medium': 35,  // $35 per day
    'High': 95    // $95 per day
};

// --- Mock ML Model Functions ---

// Mock function to simulate ML model estimating accommodation cost
function mockEstimateAccommodation(accommodationType, numberOfDays, country) {
    let baseRate = 0;
    switch (accommodationType) {
        case 'Hostel': baseRate = 20; break;
        case '3Star': baseRate = 50; break;
        case '4Star': baseRate = 90; break;
        case '5Star': baseRate = 180; break;
        case 'Airbnb': baseRate = 70; break;
        default: baseRate = 50;
    }
    // Simple mock adjustment based on country (e.g., European countries are 1.2x)
    const countryFactor = (country && country.toLowerCase().includes('india')) ? 0.8 : 1.1; 
    const cost = Math.round(baseRate * numberOfDays * countryFactor);
    return cost;
}

// Mock function to simulate ML model estimating transportation cost
function mockEstimateTransportation(transportType, distanceKm) {
    let costPerKm = 0;
    switch (transportType) {
        case 'Train': costPerKm = 0.10; break;
        case 'Bus': costPerKm = 0.08; break;
        case 'Flight': costPerKm = 0.35; break;
        case 'Car': costPerKm = 0.15; break;
        case 'Bike': costPerKm = 0.05; break;
        default: costPerKm = 0.1;
    }
    const cost = Math.round(costPerKm * distanceKm + 50); // Add a $50 base fee
    return cost;
}

// --- API Endpoint ---

app.post('/api/estimate', (req, res) => {
    const { 
        userId, 
        startLocation, 
        destination, 
        country, 
        transportType, 
        numberOfDays, 
        accommodationType, 
        foodPreference,
        activityPreference,
        distanceKm 
    } = req.body;

    // 1. Input Validation
    if (!numberOfDays || !distanceKm || !transportType || !accommodationType || !foodPreference || !activityPreference) {
        return res.status(400).json({ error: 'Missing required parameters for estimation.' });
    }

    // 2. Cost Calculation (ML Mock Sections)
    const accommodationCost = mockEstimateAccommodation(accommodationType, numberOfDays, country);
    const transportationCost = mockEstimateTransportation(transportType, distanceKm);

    // 3. Cost Calculation (Rule-Based Sections)
    const dailyFoodCost = FOOD_COSTS[foodPreference] || 0;
    const foodCost = dailyFoodCost * numberOfDays;
    
    const dailyActivityCost = ACTIVITY_COSTS[activityPreference] || 0;
    const activityCost = dailyActivityCost * numberOfDays;

    // 4. Total Cost
    const estimatedCost = accommodationCost + transportationCost + foodCost + activityCost;

    // 5. Response
    res.json({
        estimatedCost: estimatedCost.toFixed(2), // Format to 2 decimal places
        breakdown: {
            accommodation: accommodationCost.toFixed(2),
            transportation: transportationCost.toFixed(2),
            food: foodCost.toFixed(2),
            activity: activityCost.toFixed(2)
        },
        message: `Estimation complete for ${destination}. Rule-based costs applied successfully.`
    });
});

// Start the server
app.listen(port, () => {
    console.log(`Travel Estimator backend running at http://localhost:${port}`);
    console.log(`API endpoint: http://localhost:${port}/api/estimate`);
});