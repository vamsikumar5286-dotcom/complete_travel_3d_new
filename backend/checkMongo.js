// checkMongo.js
const { MongoClient } = require("mongodb");

// Replace these with your MongoDB details
const uri = "mongodb://localhost:27017"; // or your MongoDB Atlas connection string
const dbName = "travelDB"; // replace with your DB name
const collectionName = "trip_collection"; // your collection name

async function main() {
  const client = new MongoClient(uri);

  try {
    await client.connect();
    console.log("Connected to MongoDB");

    const db = client.db(dbName);
    const collection = db.collection(collectionName);

    const records = await collection.find({}).toArray();
    console.log("All stored predictions:");
    console.log(records);
  } catch (err) {
    console.error(err);
  } finally {
    await client.close();
  }
}

main();