# Webhooks and Retries

## Delivery and retries

Webhooks are delivered over HTTPS with a 10 second timeout. Failed deliveries are retried 8 times with exponential backoff: after 10 seconds, 1 minute, 5 minutes, 30 minutes, 2 hours, 6 hours, 12 hours and 24 hours. After the final attempt the event is marked failed and can be replayed manually from the dashboard for 30 days.

## Signature verification

Every request carries an X-Northwind-Signature header containing an HMAC-SHA256 of the raw body using the endpoint secret. Consumers must verify the signature before parsing the payload, and must tolerate duplicate events, since delivery is at-least-once.
