const express = require("express");
const router = express.Router();
const fetchuser = require("../middleware/fetchuser"); // Middleware for authentication
const { handleQuery, getQueryHistory } = require("../controllers/queryController");

// Handle user query and store in DB
router.post("/", fetchuser, handleQuery);

// Fetch query history for the authenticated user
router.get("/history", fetchuser, getQueryHistory);

module.exports = router;