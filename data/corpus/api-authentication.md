# API Authentication

## Keys and scopes

Requests authenticate with a bearer API key. Each key is bound to a set of scopes: read, write, admin and billing. A key never inherits more scope than the member who created it. Keys are shown once at creation and cannot be retrieved later; only the last four characters are displayed afterwards.

## Rotation

Keys can be rotated without downtime by creating a second key, deploying it, then revoking the first. Northwind recommends rotation every 90 days. Revocation takes effect within 30 seconds across all regions. Keys that appear in a public repository are revoked automatically when detected.
