const express = require('express');
const router = express.Router();
const Customer = require('../models/Customer');
const { body, validationResult } = require('express-validator');

// Validation middleware for customer data
const customerValidation = [
  body('name').trim().isLength({ min: 2, max: 100 }).withMessage('Name must be between 2 and 100 characters'),
  body('accountNumber')
    .trim()
    .matches(/^[A-Z]{2}\d{10}$/)
    .withMessage('Account number must be 2 letters followed by 10 digits'),
  body('email')
    .optional({ checkFalsy: true })
    .isEmail()
    .normalizeEmail(),
  body('mobile')
    .optional({ checkFalsy: true })
    .matches(/^(\+\d{1,3}[- ]?)?\d{10}$/)
    .withMessage('Invalid mobile number'),
  body('accountBalance')
    .isFloat({ min: 0 })
    .withMessage('Account balance must be a non-negative number')
];

// Middleware to handle validation errors
const handleValidationErrors = (req, res, next) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ 
      success: false, 
      errors: errors.array() 
    });
  }
  next();
};

// Add or Update customer
router.post(
  '/add-or-update', 
  customerValidation, 
  handleValidationErrors,
  async (req, res) => {
    try {
      const data = req.body;
      data.lastUpdated = new Date();

      // Use findOneAndUpdate with additional options for more robust upsert
      const customer = await Customer.findOneAndUpdate(
        { accountNumber: data.accountNumber },
        data,
        { 
          new: true,           // Return the modified document
          upsert: true,        // Create if not exists
          runValidators: true, // Run model validations on update
          context: 'query'     // Needed for unique validation on update
        }
      );

      res.status(200).json({ 
        success: true, 
        customer,
        message: customer ? 'Customer saved successfully' : 'Customer created successfully'
      });
    } catch (err) {
      console.error('Customer save error:', err);
      res.status(500).json({ 
        success: false, 
        message: 'Error saving customer', 
        error: err.message 
      });
    }
  }
);

// Get customer by account number
router.get('/:accountNumber', async (req, res) => {
  try {
    const customer = await Customer.findOne({ 
      accountNumber: req.params.accountNumber 
    });

    if (!customer) {
      return res.status(404).json({ 
        success: false, 
        message: 'Customer not found' 
      });
    }

    res.status(200).json({ 
      success: true, 
      customer 
    });
  } catch (err) {
    res.status(500).json({ 
      success: false, 
      message: 'Server error', 
      error: err.message 
    });
  }
});

// Add a transaction to a customer
router.post('/:accountNumber/transaction', async (req, res) => {
  try {
    const { description, amount, type } = req.body;
    
    // Validate transaction inputs
    if (!description || !amount || !type) {
      return res.status(400).json({
        success: false,
        message: 'Description, amount, and type are required'
      });
    }

    const customer = await Customer.findOne({ 
      accountNumber: req.params.accountNumber 
    });

    if (!customer) {
      return res.status(404).json({ 
        success: false, 
        message: 'Customer not found' 
      });
    }

    // Add transaction using the custom method
    await customer.addTransaction(description, amount, type);

    res.status(200).json({
      success: true,
      message: 'Transaction added successfully',
      customer
    });
  } catch (err) {
    res.status(500).json({ 
      success: false, 
      message: 'Error adding transaction', 
      error: err.message 
    });
  }
});

// Get all customers (with optional filtering)
router.get('/', async (req, res) => {
  try {
    const { 
      accountType, 
      loanStatus, 
      creditCardStatus, 
      kycStatus 
    } = req.query;

    // Build filter object
    const filter = {};
    if (accountType) filter.accountType = accountType;
    if (loanStatus) filter.loanStatus = loanStatus;
    if (creditCardStatus) filter.creditCardStatus = creditCardStatus;
    if (kycStatus) filter.kycStatus = kycStatus;

    const customers = await Customer.find(filter)
      .select('-recentTransactions') // Exclude detailed transactions
      .sort({ lastUpdated: -1 })     // Sort by most recently updated
      .limit(100);                   // Limit to prevent overwhelming response

    res.status(200).json({
      success: true,
      count: customers.length,
      customers
    });
  } catch (err) {
    res.status(500).json({ 
      success: false, 
      message: 'Error fetching customers', 
      error: err.message 
    });
  }
});

// Delete a customer by account number
router.delete('/:accountNumber', async (req, res) => {
  try {
    const customer = await Customer.findOneAndDelete({ 
      accountNumber: req.params.accountNumber 
    });

    if (!customer) {
      return res.status(404).json({ 
        success: false, 
        message: 'Customer not found' 
      });
    }

    res.status(200).json({ 
      success: true, 
      message: 'Customer deleted successfully',
      deletedCustomer: customer
    });
  } catch (err) {
    console.error('Delete customer error:', err);
    res.status(500).json({ 
      success: false, 
      message: 'Error deleting customer', 
      error: err.message 
    });
  }
});

module.exports = router;