# API Rate Limits

## Limits by plan

Rate limits are applied per workspace: Starter 10 requests per second, Team 50, Business 200, Enterprise negotiated. Bursts of up to three times the sustained rate are allowed for 10 seconds. Exceeding the limit returns HTTP 429 with a Retry-After header in seconds.

## Handling 429 responses

Clients should back off exponentially starting at one second and retry up to five times. Rate limit headers X-RateLimit-Limit, X-RateLimit-Remaining and X-RateLimit-Reset are returned on every response. Bulk imports should use the batch endpoint, which accepts up to 500 objects per request.

## Raising a limit

A temporary limit increase for a migration can be requested 5 business days in advance through the support portal. Permanent increases require a plan change.
