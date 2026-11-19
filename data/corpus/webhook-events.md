# Webhook Event Catalogue

## Event types

Available events are document.created, document.updated, document.deleted, workspace.member_added, workspace.member_removed, export.completed, billing.invoice_paid and billing.payment_failed. Each endpoint can subscribe to a subset of events; subscribing to all events is allowed but discouraged because it increases delivery volume.

## Payload envelope

Every payload contains id, type, created_at and data. The id is stable across retries of the same event. Ordering is not guaranteed, so consumers should compare created_at rather than assume arrival order.
