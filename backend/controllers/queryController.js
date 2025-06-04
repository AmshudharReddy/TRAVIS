const { processQuery } = require("../services/processors/queryProcessor");
const { processCategory } = require("../services/processors/categoryProcessor");
const { translateResponse } = require("../services/processors/translatorProcessor");
const QueryHistory = require("../models/QueryHistory");
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

// Handle Category
const handleCategory = async (req, res) => {
  console.log('Handling category request...'); // Debug: Log start of request
  try {
    // Log incoming request details
    console.log('Request body:', req.body);
    console.log('User ID from token:', req.user?.id);

    // Extract query and userId from request
    const { query } = req.body;
    const userId = req.user?.id;

    // Validate inputs
    if (!query) {
      console.warn('Validation failed: Query is missing');
      return res.status(400).json({ error: 'Query is required' });
    }
    if (!userId) {
      console.warn('Validation failed: User ID is missing or invalid');
      return res.status(401).json({ error: 'Unauthorized: Invalid or missing user token' });
    }
    if (typeof query !== 'string' || query.trim() === '') {
      console.warn('Validation failed: Query is not a valid string');
      return res.status(400).json({ error: 'Query must be a non-empty string' });
    }

    // Log query being processed
    console.log(`Processing category for query: "${query}"`);

    // Call processCategory to get the category
    const { category } = await processCategory(query);
    console.log(`Category determined: "${category}"`);

    // Validate category response
    if (!category || typeof category !== 'string') {
      console.warn('Invalid category response:', category);
      return res.status(500).json({ error: 'Failed to determine category' });
    }

    // Send successful response
    res.json({ category });
    console.log('Category response sent successfully');
  } catch (error) {
    // Log detailed error information
    console.error('Error in handleCategory:', {
      message: error.message,
      stack: error.stack,
      query: req.body.query,
      userId: req.user?.id,
    });
    res.status(500).json({ error: 'Internal Server Error', details: error.message });
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
    handleTelugu,
    handleCategory
};
