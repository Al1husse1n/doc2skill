# Webhooks and Event Notifications

Subscribe to HTTP webhook notifications to keep your local database synchronized with payment events.

## Critical Event Types

- `charge.succeeded`: Dispatched when a payment charge completes and funds are captured.
- `charge.failed`: Dispatched when a payment attempt fails or is declined.
- `invoice.paid`: Dispatched when a recurring customer invoice is successfully settled.
- `customer.subscription.deleted`: Dispatched when an active recurring subscription is canceled.

