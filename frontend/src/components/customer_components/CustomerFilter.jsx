"use client"

import { useState } from "react"
import { FaFilter, FaUndo } from "react-icons/fa"

// Predefined filter options matching the model's enums
const FILTER_OPTIONS = {
  accountType: ["Savings", "Current", "Salary", "NRI"],
  loanStatus: ["No Active Loan", "In Progress", "Approved", "Rejected"],
  creditCardStatus: ["Not Requested", "In Progress", "Delivered", "Rejected"],
  kycStatus: ["Pending", "Verified", "Rejected"],
}

const CustomerFilter = ({ onFilterChange, currentFilter }) => {
  const [localFilter, setLocalFilter] = useState(currentFilter || {})

  // Handle individual filter changes
  const handleFilterChange = (e) => {
    const { name, value } = e.target
    const newFilter = {
      ...localFilter,
      [name]: value || undefined, // Use undefined to remove filter if empty
    }

    // Remove undefined/empty filters
    const cleanedFilter = Object.fromEntries(Object.entries(newFilter).filter(([, v]) => v !== undefined))

    setLocalFilter(cleanedFilter)
    onFilterChange(cleanedFilter)
  }

  // Reset all filters
  const handleResetFilters = () => {
    setLocalFilter({})
    onFilterChange({})
  }

  return (
    <div className="customer-filter">
      <h3 className="filter-title">
        <FaFilter /> Filter Customers
      </h3>

      <div className="filter-grid">
        {/* Account Type Filter */}
        <div className="filter-group">
          <label className="filter-label">Account Type</label>
          <select
            name="accountType"
            value={localFilter.accountType || ""}
            onChange={handleFilterChange}
            className="filter-control"
          >
            <option value="">All Types</option>
            {FILTER_OPTIONS.accountType.map((type) => (
              <option key={type} value={type}>
                {type}
              </option>
            ))}
          </select>
        </div>

        {/* Loan Status Filter */}
        <div className="filter-group">
          <label className="filter-label">Loan Status</label>
          <select
            name="loanStatus"
            value={localFilter.loanStatus || ""}
            onChange={handleFilterChange}
            className="filter-control"
          >
            <option value="">All Statuses</option>
            {FILTER_OPTIONS.loanStatus.map((status) => (
              <option key={status} value={status}>
                {status}
              </option>
            ))}
          </select>
        </div>

        {/* Credit Card Status Filter */}
        <div className="filter-group">
          <label className="filter-label">Credit Card Status</label>
          <select
            name="creditCardStatus"
            value={localFilter.creditCardStatus || ""}
            onChange={handleFilterChange}
            className="filter-control"
          >
            <option value="">All Statuses</option>
            {FILTER_OPTIONS.creditCardStatus.map((status) => (
              <option key={status} value={status}>
                {status}
              </option>
            ))}
          </select>
        </div>

        {/* KYC Status Filter */}
        <div className="filter-group">
          <label className="filter-label">KYC Status</label>
          <select
            name="kycStatus"
            value={localFilter.kycStatus || ""}
            onChange={handleFilterChange}
            className="filter-control"
          >
            <option value="">All Statuses</option>
            {FILTER_OPTIONS.kycStatus.map((status) => (
              <option key={status} value={status}>
                {status}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Reset Filters Button */}
      {Object.keys(localFilter).length > 0 && (
        <div className="filter-actions">
          <button type="button" onClick={handleResetFilters} className="reset-filter-btn">
            <FaUndo /> Reset Filters
          </button>
        </div>
      )}
    </div>
  )
}

export default CustomerFilter;