const express = require("express");
const router = express.Router();
const { processQuery } = require("../services/queryProcessor");
const QueryHistory = require("../models/QueryHistory");
const fetchuser = require("../middleware/fetchuser"); // Middleware for authentication

// Handle user query and store in DB
router.post("/", fetchuser, async (req, res) => {
    try {
        const { query } = req.body;
        const userId = req.user.id; // Extract user ID from token

        if (!query) return res.status(400).json({ error: "Query is required" });

        // Get AI response (Mocked for now)
        const response = await processQuery(query);

        // Save query-response to MongoDB linked to user
        const historyEntry = new QueryHistory({ userId, query, response });
        await historyEntry.save();

        res.json({ response });
    } catch (error) {
        console.error("Error processing query:", error);
        res.status(500).json({ error: "Internal Server Error" });
    }
});

// Fetch query history for the authenticated user
router.get("/history", fetchuser, async (req, res) => {
    try {
        const userId = req.user.id;
        const history = await QueryHistory.find({ userId }).sort({ createdAt: -1 }); // Fetch only user-specific history

        res.json(history);
    } catch (error) {
        console.error("Error fetching query history:", error);
        res.status(500).json({ error: "Internal Server Error" });
    }
});

module.exports = router;
