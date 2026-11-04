---
title: Customer Order Service
type: sd_overview
service: customer-order
---

# Customer Order Service

The Customer Order service orchestrates customer orders from checkout to delivery.

## Overview

This service handles all aspects of customer order management including
creation, payment, shipping, and returns.

### Key Capabilities

- Account lifecycle management (open, close, dormant)
- Transaction processing (credit, debit)
- Interest calculation and application
- Fee management

### Behavior Qualifiers

The following behavior qualifiers are supported:

- DepositsAndWithdrawals
- Payments
- Interest
- ServiceFees
- AccountLien
- AccountSweep

## API Operations

The service exposes the following REST operations.

### Initiate

Create a new customer order facility arrangement.

POST /CustomerOrder/Initiate

### Retrieve

Get details about an existing customer order.

GET /CustomerOrder/{id}/Retrieve

## Connected Services

- Payment Initiation
- Payment Execution
- Financial Accounting
- Compliance Reporting
