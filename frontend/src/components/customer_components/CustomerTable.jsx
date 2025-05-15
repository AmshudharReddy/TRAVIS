"use client"

import { useState } from "react"
import { FaEdit, FaTrash, FaExclamationTriangle } from "react-icons/fa"

const CustomerTable = ({ data, isLoading, onEdit, onDelete }) => {
  const [confirmDelete, setConfirmDelete] = useState(null)

  // Format currency
  const formatCurrency = (amount) => {
    return new Intl.NumberFormat("en-IN", {
      style: "currency",
      currency: "INR",
    }).format(amount)
  }

  // Determine status class
  const getStatusClass = (status) => {
    switch (status) {
      case "Approved":
      case "Delivered":
      case "Verified":
        return "status-approved"
      case "In Progress":
        return "status-in-progress"
      case "Rejected":
      case "Pending":
        return "status-rejected"
      case "Not Requested":
      case "No Active Loan":
        return "status-not-requested"
      default:
        return ""
    }
  }

  // Handle delete confirmation
  const handleDeleteConfirmation = (customer) => {
    setConfirmDelete(customer)
  }

  // Confirm delete
  const confirmDeleteCustomer = () => {
    if (confirmDelete) {
      onDelete(confirmDelete.accountNumber)
      setConfirmDelete(null)
    }
  }

  // Cancel delete
  const cancelDelete = () => {
    setConfirmDelete(null)
  }

  // Render loading state
  if (isLoading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p className="loading-text">Loading customers...</p>
      </div>
    )
  }

  // Render empty state
  if (!data || data.length === 0) {
    return (
      <div className="empty-container">
        <FaExclamationTriangle size={40} color="#6b7280" />
        <p className="empty-text">No customers found</p>
      </div>
    )
  }

  return (
    <div className="customer-table-container">
      <div className="overflow-x-auto">
        <table className="customer-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Account No.</th>
              <th>Account Type</th>
              <th>Balance</th>
              <th>Loan Status</th>
              <th>Credit Card</th>
              <th>KYC Status</th>
              <th className="text-center">Actions</th>
            </tr>
          </thead>
          <tbody>
            {data.map((customer) => (
              <tr key={customer.accountNumber}>
                <td>
                  <div className="font-medium">{customer.name}</div>
                  {customer.email && <div className="text-xs text-gray-500">{customer.email}</div>}
                </td>
                <td>{customer.accountNumber}</td>
                <td>{customer.accountType}</td>
                <td>{formatCurrency(customer.accountBalance)}</td>
                <td>
                  <span className={`status-badge ${getStatusClass(customer.loanStatus)}`}>{customer.loanStatus}</span>
                </td>
                <td>
                  <span className={`status-badge ${getStatusClass(customer.creditCardStatus)}`}>
                    {customer.creditCardStatus}
                  </span>
                </td>
                <td>
                  <span className={`status-badge ${getStatusClass(customer.kycStatus)}`}>{customer.kycStatus}</span>
                </td>
                <td>
                  <div className="table-actions">
                    <button onClick={() => onEdit(customer)} className="btn-edit">
                      <FaEdit className="btn-icon" /> Edit
                    </button>
                    <button onClick={() => handleDeleteConfirmation(customer)} className="btn-delete">
                      <FaTrash className="btn-icon" /> Delete
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Delete Confirmation Modal */}
      {confirmDelete && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-header">
              <h3 className="modal-title">Confirm Deletion</h3>
            </div>
            <div className="modal-body">
              <p>
                Are you sure you want to delete the customer <strong>{confirmDelete.name}</strong> with account number{" "}
                <strong>{confirmDelete.accountNumber}</strong>?
              </p>
              <p className="text-red-500 mt-4 text-sm">
                <FaExclamationTriangle className="inline-block mr-2" />
                This action cannot be undone.
              </p>
            </div>
            <div className="modal-footer">
              <button onClick={cancelDelete} className="reset-filter-btn">
                Cancel
              </button>
              <button onClick={confirmDeleteCustomer} className="btn-delete">
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default CustomerTable;