# Error Handling and Decline Codes

All API errors return standard JSON payloads with an `error` dictionary containing `type`, `code`, and `message`.

## Common Error Codes

- `card_declined`: The card could not be charged due to insufficient funds, expiration, or anti-fraud rules. Prompt the customer for an alternate payment method.
- `rate_limit_exceeded`: Too many requests hit the API in a short window. Implement exponential backoff.
- `invalid_request_error`: Missing required parameters, such as currency or amount.
- `authentication_error`: Invalid or expired API credentials.

