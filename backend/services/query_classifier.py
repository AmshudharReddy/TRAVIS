from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load transformer model (BART for zero-shot classification)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Banking intents list
INTENT_CATEGORIES = [
    "Check Account Balance",
    "Download Bank Statement",
    "Transaction History",
    "Fund Transfer",
    "Loan Inquiry & Application",
    "Credit Card Services",
    "Debit Card Services",
    "Card Blocking",
    "Card Activation",
    "Report Fraudulent Activity",
    "UPI Payments",
    "Net Banking Issues",
    "Loan Repayment & EMI",
    "Cheque Services",
    "Open a New Account",
    "Close Bank Account",
    "Check Interest Rates",
    "Fixed Deposit Inquiry",
    "Recurring Deposit Inquiry",
    "Reset Password",
    "Customer Service",
    "Find Nearest Branch"
]

@app.route('/classify', methods=['POST'])
def classify_query():
    data = request.json
    user_query = data.get("query", "")

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # Classify the query
    result = classifier(user_query, INTENT_CATEGORIES)
    best_intent = result["labels"][0]  # Most confident classification

    return jsonify({
        "query": user_query,
        "predicted_intent": best_intent
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
