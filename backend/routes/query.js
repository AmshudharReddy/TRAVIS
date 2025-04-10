const express = require("express");
const router = express.Router();
const fetchuser = require("../middleware/fetchuser"); // Middleware for authentication
const { handleQuery,translate, getQueryHistory } = require("../controllers/queryController");

// Handle user query and store in DB
router.post("/", fetchuser, handleQuery);
router.post("/translate",fetchuser,translate);

// Fetch query history for the authenticated user
router.get("/history", fetchuser, getQueryHistory);

module.exports = router;