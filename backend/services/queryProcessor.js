const axios = require("axios");

async function processQuery(query) {
    try {
        // Mock AI Response (Until Transformer Model is Ready)
        const modelResponse = {
            query: query,  // Logging the query
            response: `AI generated response for query: ${query}`
        };

        return modelResponse.response; // Correctly return response
    } catch (error) {
        console.error("Error in transformer model:", error);
        return "Sorry, I couldn't process that query.";
    }
}


module.exports = { processQuery };
