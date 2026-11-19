# Pagination and Filtering

## Cursors

List endpoints return up to 100 objects per page and use cursor pagination, not offsets, because offsets drift when records are inserted concurrently. The response contains a next_cursor field; pass it as the cursor parameter to fetch the next page. A null cursor means the end of the result set.

## Filtering

Every list endpoint accepts created_after, created_before and updated_after as ISO 8601 timestamps. Filters combine with AND. Sorting is limited to created_at and updated_at; arbitrary sort fields are not supported.
