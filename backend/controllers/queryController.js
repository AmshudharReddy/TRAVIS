const { processQuery } = require("../services/queryProcessor");
const QueryHistory = require("../models/QueryHistory");
const { translateResponse } = require("../services/translatorProcessor");
const axios = require("axios");

// Controller for processing user queries
const handleQuery = async (req, res) => {
    try {
        const { query } = req.body;
        const userId = req.user.id;

        if (!query) return res.status(400).json({ error: "Query is required" });

        // Process query through transformer model
        const { category, response } = await processQuery(query);

        // Save to MongoDB
        const historyEntry = new QueryHistory({ userId, query, category, response });
        const savedEntry = await historyEntry.save();

        // Return model response + history ID for future reference
        res.json({ category, response, historyId: savedEntry._id });
    } catch (error) {
        console.error("Error processing query:", error);
        res.status(500).json({ error: "Internal Server Error" });
    }
};

// Controller for translating model response
const handleTranslate = async (req, res) => {
    try {
        const { response, historyId } = req.body;
        const userId = req.user.id;

        if (!response) return res.status(400).json({ error: "Response is required" });

        const { translation } = await translateResponse(response);

        // Optional: update translated response in existing query history
        if (historyId) {
            await QueryHistory.findOneAndUpdate(
                { _id: historyId, userId },
                { translatedResponse: translation }
            );
        }

        res.json({ translation });
    } catch (error) {
        console.error("Error translating response:", error);
        res.status(500).json({ error: "Internal Server Error" });
    }
};

// Controller for fetching user-specific query history
const getQueryHistory = async (req, res) => {
    try {
        const userId = req.user.id;
        const history = await QueryHistory.find({ userId }).sort({ createdAt: -1 });
        res.json(history);
    } catch (error) {
        console.error("Error fetching query history:", error);
        res.status(500).json({ error: "Internal Server Error" });
    }
};

// Controller for converting text to Telugu speech (TTS)
const handleTelugu = async (req, res) => {
    try {
        const response = await axios.post(
            'http://127.0.0.1:5001/api/tts',
            { text: req.body.text },
            { responseType: 'stream' }
        );

        res.setHeader('Content-Type', 'audio/mpeg');
        response.data.pipe(res);
    } catch (err) {
        console.error("Error generating speech:", err);
        res.status(500).send('Error generating speech');
    }
};

module.exports = {
    handleQuery,
    handleTranslate,
    getQueryHistory,
    handleTelugu
};
