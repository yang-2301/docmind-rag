# Data Export and Portability

## Formats

Workspace data can be exported as JSON Lines or CSV. Documents are exported in their original format together with a manifest file that records metadata. Exports are produced asynchronously and a download link is emailed when ready; the link expires after 72 hours.

## Limits

A single export is capped at 500 GB. Larger workspaces must be exported in partitions by date range. Export generation usually takes under 2 hours for 100 GB and is not counted against API rate limits.
