"use client"

import { useState, useEffect } from "react"
import { FaUser, FaAddressCard, FaCreditCard, FaMoneyBillWave, FaShieldAlt, FaBook, FaTimes } from "react-icons/fa"

// Predefined options for dropdowns
const ACCOUNT_TYPES = ["Savings", "Current", "Salary", "NRI"]
const CREDIT_CARD_STATUSES = ["Not Requested", "In Progress", "Delivered", "Rejected"]
const CHEQUE_BOOK_STATUSES = ["Not Requested", "In Progress", "Delivered", "Rejected"]
const LOAN_STATUSES = ["No Active Loan", "In Progress", "Approved", "Rejected"]
const LOAN_TYPES = ["Home", "Personal", "Auto", "Education"]
const KYC_STATUSES = ["Pending", "Verified", "Rejected"]

const CustomerForm = ({ onSubmit, initialData, onCancel }) => {
  // Initial state with comprehensive fields and default values
  const [formData, setFormData] = useState({
    name: "",
    accountNumber: "",
    email: "",
    mobile: "",
    accountType: "Savings",
    accountBalance: "",

    // Credit Card Section
    creditCardStatus: "Not Requested",
    creditCardLimit: "",
    creditCardFeatures: [],

    // Cheque Book Section
    chequeBookStatus: "Not Requested",

    // Loan Section
    loanStatus: "No Active Loan",
    loanType: "",
    loanAmount: "",
    loanEMI: "",

    // KYC Section
    kycStatus: "Pending",
    securityAlerts: [],
  })

  // Reset form when initial data changes
  useEffect(() => {
    if (initialData) {
      // Create a new object with fallback to default values
      const transformedData = {
        name: initialData.name || "",
        accountNumber: initialData.accountNumber || "",
        email: initialData.email || "",
        mobile: initialData.mobile || "",
        accountType: initialData.accountType || "Savings",
        accountBalance: initialData.accountBalance ? initialData.accountBalance.toString() : "",

        // Credit Card Section
        creditCardStatus: initialData.creditCardStatus || "Not Requested",
        creditCardLimit: initialData.creditCardLimit ? initialData.creditCardLimit.toString() : "",
        creditCardFeatures: initialData.creditCardFeatures || [],

        // Cheque Book Section
        chequeBookStatus: initialData.chequeBookStatus || "Not Requested",

        // Loan Section
        loanStatus: initialData.loanStatus || "No Active Loan",
        loanType: initialData.loanType || "",
        loanAmount: initialData.loanAmount ? initialData.loanAmount.toString() : "",
        loanEMI: initialData.loanEMI ? initialData.loanEMI.toString() : "",

        // KYC Section
        kycStatus: initialData.kycStatus || "Pending",
        securityAlerts: initialData.securityAlerts || [],
      }

      setFormData(transformedData)
    } else {
      // Reset to initial state when no initial data
      setFormData({
        name: "",
        accountNumber: "",
        email: "",
        mobile: "",
        accountType: "Savings",
        accountBalance: "",
        creditCardStatus: "Not Requested",
        creditCardLimit: "",
        creditCardFeatures: [],
        chequeBookStatus: "Not Requested",
        loanStatus: "No Active Loan",
        loanType: "",
        loanAmount: "",
        loanEMI: "",
        kycStatus: "Pending",
        securityAlerts: [],
      })
    }
  }, [initialData])

  // Handle input changes
  const handleChange = (e) => {
    const { name, value, type, checked } = e.target

    // Special handling for different input types
    if (type === "checkbox") {
      // Handle credit card features
      if (name === "creditCardFeatures") {
        setFormData((prev) => {
          const features = checked
            ? [...prev.creditCardFeatures, value]
            : prev.creditCardFeatures.filter((f) => f !== value)
          return { ...prev, creditCardFeatures: features }
        })
      }
      // Handle security alerts
      else if (name === "securityAlerts") {
        setFormData((prev) => {
          const alerts = checked ? [...prev.securityAlerts, value] : prev.securityAlerts.filter((a) => a !== value)
          return { ...prev, securityAlerts: alerts }
        })
      }
    } else {
      // Standard input handling
      setFormData((prev) => ({
        ...prev,
        [name]: value,
      }))
    }
  }

  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault()

    // Validate required fields
    if (!formData.name || !formData.accountNumber) {
      alert("Name and Account Number are required")
      return
    }

    // Convert string inputs to appropriate types
    const processedData = {
      ...formData,
      accountBalance: formData.accountBalance ? Number.parseFloat(formData.accountBalance) : 0,
      creditCardLimit: formData.creditCardLimit ? Number.parseFloat(formData.creditCardLimit) : undefined,
      loanAmount: formData.loanAmount ? Number.parseFloat(formData.loanAmount) : undefined,
      loanEMI: formData.loanEMI ? Number.parseFloat(formData.loanEMI) : undefined,
    }

    // Remove empty or undefined fields
    Object.keys(processedData).forEach((key) => {
      if (processedData[key] === "" || processedData[key] === undefined) {
        delete processedData[key]
      }
    })

    // Ensure loan-related fields are consistent
    if (processedData.loanStatus === "No Active Loan") {
      delete processedData.loanType
      delete processedData.loanAmount
      delete processedData.loanEMI
    }

    // Submit processed data
    onSubmit(processedData)
  }

  return (
    <div className="customer-form-container">
      <div className="customer-form-header" style={{ position: "relative" }}>
        {initialData ? "Edit Customer" : "Add New Customer"}
        <button
          onClick={onCancel}
          className="cancel-form-btn"
          style={{
            position: "absolute",
            right: "15px",
            top: "50%",
            transform: "translateY(-50%)",
            background: "rgba(255, 255, 255, 0.2)",
            border: "none",
            borderRadius: "50%",
            width: "32px",
            height: "32px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            cursor: "pointer",
            color: "white",
            transition: "all 0.2s ease",
          }}
          aria-label="Cancel"
        >
          <FaTimes />
        </button>
      </div>
      <form onSubmit={handleSubmit} className="customer-form-content">
        {/* Basic Information Section */}
        <div className="form-section">
          <h3 className="form-section-title">
            <FaUser /> Basic Information
          </h3>
          <div className="form-grid">
            <div className="form-group">
              <label className="form-label">
                Name <span className="text-red-500">*</span>
              </label>
              <input
                className="form-control"
                name="name"
                value={formData.name}
                onChange={handleChange}
                placeholder="Full Name"
                required
              />
            </div>

            <div className="form-group">
              <label className="form-label">
                Account Number <span className="text-red-500">*</span>
              </label>
              <input
                className="form-control"
                name="accountNumber"
                value={formData.accountNumber}
                onChange={handleChange}
                placeholder="e.g. AB1234567890"
                required
                pattern="[A-Z]{2}\d{10}"
                title="Account number must be 2 letters followed by 10 digits"
              />
            </div>

            <div className="form-group">
              <label className="form-label">Email</label>
              <input
                type="email"
                className="form-control"
                name="email"
                value={formData.email}
                onChange={handleChange}
                placeholder="Email Address"
              />
            </div>

            <div className="form-group">
              <label className="form-label">Mobile Number</label>
              <input
                type="tel"
                className="form-control"
                name="mobile"
                value={formData.mobile}
                onChange={handleChange}
                placeholder="Mobile Number"
                pattern="\+?[0-9]{10,14}"
                title="Mobile number must be 10-14 digits"
              />
            </div>
          </div>
        </div>

        {/* Account Details Section */}
        <div className="form-section">
          <h3 className="form-section-title">
            <FaAddressCard /> Account Details
          </h3>
          <div className="form-grid">
            <div className="form-group">
              <label className="form-label">Account Type</label>
              <select className="form-control" name="accountType" value={formData.accountType} onChange={handleChange}>
                {ACCOUNT_TYPES.map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Account Balance (₹)</label>
              <input
                type="number"
                className="form-control"
                name="accountBalance"
                value={formData.accountBalance}
                onChange={handleChange}
                placeholder="0.00"
                min="0"
                step="0.01"
              />
            </div>
          </div>
        </div>

        {/* Credit Card Section */}
        <div className="form-section">
          <h3 className="form-section-title">
            <FaCreditCard /> Credit Card Details
          </h3>
          <div className="form-grid">
            <div className="form-group">
              <label className="form-label">Credit Card Status</label>
              <select
                className="form-control"
                name="creditCardStatus"
                value={formData.creditCardStatus}
                onChange={handleChange}
              >
                {CREDIT_CARD_STATUSES.map((status) => (
                  <option key={status} value={status}>
                    {status}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Credit Card Limit (₹)</label>
              <input
                type="number"
                className="form-control"
                name="creditCardLimit"
                value={formData.creditCardLimit}
                onChange={handleChange}
                placeholder="0.00"
                min="0"
                step="0.01"
                disabled={formData.creditCardStatus === "Not Requested" || formData.creditCardStatus === "Rejected"}
              />
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">Credit Card Features</label>
            <div className="checkbox-group">
              {["Cashback", "Travel Rewards", "Airport Lounge", "No Annual Fee"].map((feature) => (
                <label key={feature} className="checkbox-item">
                  <input
                    type="checkbox"
                    name="creditCardFeatures"
                    value={feature}
                    checked={formData.creditCardFeatures.includes(feature)}
                    onChange={handleChange}
                    disabled={formData.creditCardStatus === "Not Requested" || formData.creditCardStatus === "Rejected"}
                  />
                  <span>{feature}</span>
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* Cheque Book Section */}
        <div className="form-section">
          <h3 className="form-section-title">
            <FaBook /> Cheque Book
          </h3>
          <div className="form-grid">
            <div className="form-group">
              <label className="form-label">Cheque Book Status</label>
              <select
                className="form-control"
                name="chequeBookStatus"
                value={formData.chequeBookStatus}
                onChange={handleChange}
              >
                {CHEQUE_BOOK_STATUSES.map((status) => (
                  <option key={status} value={status}>
                    {status}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Loan Section */}
        <div className="form-section">
          <h3 className="form-section-title">
            <FaMoneyBillWave /> Loan Details
          </h3>
          <div className="form-grid">
            <div className="form-group">
              <label className="form-label">Loan Status</label>
              <select className="form-control" name="loanStatus" value={formData.loanStatus} onChange={handleChange}>
                {LOAN_STATUSES.map((status) => (
                  <option key={status} value={status}>
                    {status}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Loan Type</label>
              <select
                className="form-control"
                name="loanType"
                value={formData.loanType}
                onChange={handleChange}
                disabled={formData.loanStatus === "No Active Loan" || formData.loanStatus === "Rejected"}
              >
                <option value="">Select Loan Type</option>
                {LOAN_TYPES.map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Loan Amount (₹)</label>
              <input
                type="number"
                className="form-control"
                name="loanAmount"
                value={formData.loanAmount}
                onChange={handleChange}
                placeholder="0.00"
                min="0"
                step="0.01"
                disabled={formData.loanStatus === "No Active Loan" || formData.loanStatus === "Rejected"}
              />
            </div>

            <div className="form-group">
              <label className="form-label">Monthly EMI (₹)</label>
              <input
                type="number"
                className="form-control"
                name="loanEMI"
                value={formData.loanEMI}
                onChange={handleChange}
                placeholder="0.00"
                min="0"
                step="0.01"
                disabled={formData.loanStatus === "No Active Loan" || formData.loanStatus === "Rejected"}
              />
            </div>
          </div>
        </div>

        {/* KYC and Security Section */}
        <div className="form-section">
          <h3 className="form-section-title">
            <FaShieldAlt /> KYC & Security
          </h3>
          <div className="form-grid">
            <div className="form-group">
              <label className="form-label">KYC Status</label>
              <select className="form-control" name="kycStatus" value={formData.kycStatus} onChange={handleChange}>
                {KYC_STATUSES.map((status) => (
                  <option key={status} value={status}>
                    {status}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">Security Alerts</label>
            <div className="checkbox-group">
              {["Password Changed", "Login from New Device", "Suspicious Activity"].map((alert) => (
                <label key={alert} className="checkbox-item">
                  <input
                    type="checkbox"
                    name="securityAlerts"
                    value={alert}
                    checked={formData.securityAlerts.includes(alert)}
                    onChange={handleChange}
                  />
                  <span>{alert}</span>
                </label>
              ))}
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="form-actions">
          <button type="button" onClick={onCancel} className="reset-filter-btn">
            Cancel
          </button>
          <button type="submit" className="add-customer-btn">
            {initialData ? "Update Customer" : "Add Customer"}
          </button>
        </div>
      </form>
    </div>
  )
}

export default CustomerForm;