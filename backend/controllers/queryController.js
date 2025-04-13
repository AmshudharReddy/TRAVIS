const { processQuery } = require("../services/queryProcessor");
const QueryHistory = require("../models/QueryHistory");
const { translateResponse } = require("../services/translatorProcessor");

// Controller for processing user queries
const handleQuery = async (req, res) => {
    try {
        const { query } = req.body;
        const userId = req.user.id; // Extract user ID from token

        if (!query) return res.status(400).json({ error: "Query is required" });

        // Get AI response (Mocked for now)
        const {category,response} = await processQuery(query);

        // Save query-response to MongoDB linked to user
        const historyEntry = new QueryHistory({ userId, query, category,response });
        await historyEntry.save();

        res.json({ category,response });
    } catch (error) {
        console.error("Error processing query:", error);
        res.status(500).json({ error: "Internal Server Error" });
    }
};

// Controller for translating Model Response
const handleTranslate = async (req, res) => {
    try {
        const { response } = req.body;
        const userId = req.user.id; // Extract user ID from token

        if (!response) return res.status(400).json({ error: "Response is required" });

        // Get translateed response
        const {translation} = await translateResponse(response);

        res.json({ translation});
    } catch (error) {
        console.error("Error processing query:", error);
        res.status(500).json({ error: "Internal Server Error" });
    }
};

// Controller for fetching query history
const getQueryHistory = async (req, res) => {
    try {
        const userId = req.user.id;
        const history = await QueryHistory.find({ userId }).sort({ createdAt: -1 }); // Fetch only user-specific history

        res.json(history);
    } catch (error) {
        console.error("Error fetching query history:", error);
        res.status(500).json({ error: "Internal Server Error" });
    }
};

module.exports = { handleQuery, handleTranslate, getQueryHistory };