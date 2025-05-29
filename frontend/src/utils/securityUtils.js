/**
 * Security Utility Functions
 * This module contains security-related helpers for the banking application
 */

const sensitiveKeywords = [
  // Account related
  "account balance",
  "balance enquiry",
  "check balance",
  "available balance",
  "account details",
  "account information",
  "account statement",
  "statement",

  // Card related
  "card delivery",
  "credit card",
  "credit card delivery",
  "credit card status",
  "card limit",
  "credit limit",
  "increase limit",
  "card block",
  "debit card",
  "card activation",
  "card pin",
  "card otp",

  // Loan related
  "loan",
  "loan approval",
  "loan status",
  "loan details",
  "emi details",
  "approved loan",
  "loan application",
  "interest rate",
  "foreclose loan",
  "loan statement",
  "loan account",
  "loan balance",
  "loan amount",

  // Transaction related
  "transaction history",
  "recent transactions",
  "money transfer",
  "transaction details",
  "payment history",
  "statement",
  "fund transfer",
  "neft",
  "rtgs",
  "imps",
  "upi",
  "last transaction",

  // KYC related
  "kyc",
  "kyc status",
  "know your customer",
  "verification status",
  "identity verification",
  "document verification",
  "aadhar",
  "pan card",

  // Cheque related
  "cheque book",
  "cheque status",
  "checkbook request",
  "stop payment",
  "cheque bounce",
  "cheque deposit",
  "cheque clearance",

  // Security related
  "password",
  "pin",
  "otp",
  "authentication",
  "two factor",
  "2fa",
  "security question",
  "security alert",
  "account lock",
  "reset password",
]

/**
 * Check if text contains sensitive information
 * @param {string} text - Text to check for sensitive information
 * @returns {boolean} - True if contains sensitive info, false otherwise
 */
const containsSensitiveInfo = (text) => {
  if (!text) return false

  const lowerText = text.toLowerCase()
  return sensitiveKeywords.some((keyword) => lowerText.includes(keyword))
}

/**
 * Validate account number format
 * @param {string} accountNumber - Account number to validate
 * @returns {boolean} - True if valid, false otherwise
 */
const isValidAccountNumber = (accountNumber) => {
  return /^[A-Z]{2}\d{10}$/.test(accountNumber)
}

/**
 * Mask sensitive data for display or logging
 * @param {string} data - Sensitive data to mask
 * @param {string} type - Type of data (account, email, etc.)
 * @returns {string} - Masked data
 */
const maskSensitiveData = (data, type = "default") => {
  if (!data) return ""

  switch (type.toLowerCase()) {
    case "account":
    case "accountnumber":
      // Format: XX****1234
      return data.substring(0, 2) + "****" + data.slice(-4)

    case "email":
      // Format: us***@example.com
      const [username, domain] = data.split("@")
      if (!domain) return data
      return username.substring(0, 2) + "***@" + domain

    case "phone":
    case "mobile":
      // Format: ***-***-1234
      return "***-***-" + data.slice(-4)

    default:
      // Generic masking: show first 2 and last 2 chars
      if (data.length <= 4) return "****"
      return data.substring(0, 2) + "*".repeat(data.length - 4) + data.slice(-2)
  }
}

// Export all security utilities
export { sensitiveKeywords, containsSensitiveInfo, isValidAccountNumber, maskSensitiveData }
