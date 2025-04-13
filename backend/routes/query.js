const express = require("express");
const router = express.Router();
const fetchuser = require("../middleware/fetchuser"); // Middleware for authentication
const { handleQuery, handleTranslate, getQueryHistory } = require("../controllers/queryController");

// Handle user query and store in DB
router.post("/", fetchuser, handleQuery);

// Handle response translation into local language (Here, Telugu)
router.post("/translate", fetchuser, handleTranslate);

// Fetch query history for the authenticated user
router.get("/history", fetchuser, getQueryHistory);

module.exports = router;