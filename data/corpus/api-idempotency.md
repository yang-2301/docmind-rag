# Idempotent Writes

## Idempotency keys

POST requests accept an Idempotency-Key header of up to 255 characters. A repeated request with the same key returns the original response instead of creating a second object. Keys are retained for 24 hours; reusing a key with a different payload returns 409.
