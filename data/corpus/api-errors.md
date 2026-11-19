# API Error Codes

## Status codes

400 means a malformed request or failed validation and includes a field-level error list. 401 means a missing or invalid key. 403 means the key lacks the required scope. 404 means the object does not exist or is not visible to the key. 409 means a conflict, most often a duplicate idempotency key. 422 means the payload parsed but violated a business rule. 429 is rate limiting. 5xx are server faults and are always safe to retry with backoff.

## Error body

Errors return a JSON body with code, message and request_id. Support requests without a request_id are slower to investigate, so log it on every failure.
