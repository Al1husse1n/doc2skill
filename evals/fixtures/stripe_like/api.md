# Stripe-like Payments API Reference

The Payments API provides programmatic access to card processing, customer management, and subscriptions.

## Core Endpoints

### Charges
- `POST /v1/charges`: Create a new charge with an authorized card token or payment method.
- `GET /v1/charges/{id}`: Retrieve details of an existing charge, including balance transaction and refund status.
- `POST /v1/charges/{id}/capture`: Capture an authorized payment.

### Customers
- `POST /v1/customers`: Create a new customer profile with billing email and default payment source.
- `GET /v1/customers`: List and search customer records.
- `GET /v1/customers/{id}`: Retrieve a specific customer record and active subscriptions.
- `DELETE /v1/customers/{id}`: Permanently delete a customer record.

