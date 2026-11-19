# Backup Policy

## Backup schedule

Primary storage is snapshotted every 6 hours, retained for 35 days. Weekly full backups are retained for 12 weeks offline. Backups are encrypted with a separate key from production data. Customers cannot restore snapshots themselves on Starter and Team; Business and Enterprise can self-restore the last 7 days from the admin panel.

## Restore requests

A restore request older than 7 days must be opened as a P2 ticket and takes up to 8 business hours. Restores are performed into a new workspace by default so that the current data is not overwritten.
