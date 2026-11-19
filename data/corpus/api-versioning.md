# API Versioning and Deprecation

## Versioning

The API is versioned in the path, for example /v2/documents. A new major version is released at most once a year. Backwards-incompatible changes never ship inside an existing version.

## Deprecation

Deprecated endpoints keep working for at least 12 months after the announcement and return a Deprecation header with the sunset date. Customers are notified by email and through the changelog at least 90 days before the sunset date.
