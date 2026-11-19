# Core Data Model

## Objects

A workspace contains collections; a collection contains documents; a document has revisions and tags. Tags are workspace-scoped, not global. Deleting a collection moves its documents to the workspace root rather than deleting them.

## Identifiers

All objects use prefixed identifiers: ws_ for workspaces, doc_ for documents, col_ for collections. Identifiers are stable and never reused after deletion.
