#!/usr/bin/env python3
"""Generate the demo corpus: 24 English policy documents for a fictional
B2B SaaS company ("Northwind Cloud"), plus a golden question set.

Why a generated corpus and not a scraped one:
  * reproducible — every number in the README can be re-derived;
  * no licensing or privacy issues in a public repo;
  * the questions can be designed to stress the exact failure modes the
    retriever is supposed to fix (paraphrase, identifier lookup,
    cross-section synthesis).

Run:  python3 scripts/make_corpus.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"
GOLDEN = ROOT / "data" / "golden.jsonl"

DOCS = [
    ("pricing-plans", "Plans and Included Usage", [
        ("Plan tiers",
         "Northwind Cloud is sold in four plans: Starter, Team, Business and Enterprise. "
         "Starter includes 2 seats, 50 GB of document storage and 1,000 API calls per month. "
         "Team includes 10 seats, 500 GB and 25,000 API calls. Business includes 50 seats, "
         "2 TB and 250,000 API calls. Enterprise is quoted per contract and has no fixed "
         "seat cap. Seats and storage can be raised on any plan; API calls above the "
         "included amount are billed at overage rates."),
        ("Overage pricing",
         "Overage is charged per 1,000 calls. Starter pays 0.90 USD per 1,000 calls, Team "
         "pays 0.60 USD, Business pays 0.35 USD and Enterprise pays the rate in the order "
         "form. Storage overage is 0.08 USD per GB per month on every plan except "
         "Enterprise. Overage is calculated at the end of the billing period and never "
         "blocks API traffic; the account is not throttled for exceeding included usage."),
    ]),
    ("billing-cycle", "Billing Cycle and Invoicing", [
        ("When you are charged",
         "Subscriptions renew on the first day of each calendar month, regardless of the "
         "day the contract started. New accounts are prorated from the activation date to "
         "the end of that month and then billed in full. Annual contracts are invoiced "
         "upfront and renew on the anniversary of the signature date."),
        ("Invoices and payment methods",
         "Invoices are issued on the second business day of the month and are payable "
         "within 30 days. Accepted payment methods are credit card, ACH transfer and wire. "
         "Wire transfers under 1,000 USD incur a 25 USD handling fee. Invoices more than "
         "15 days overdue suspend write access to the workspace; invoices more than 45 "
         "days overdue are referred to collections and the workspace is set to read-only."),
    ]),
    ("refunds", "Refunds and Cancellations", [
        ("Cancelling a subscription",
         "You may cancel at any time from Settings, Billing. Cancellation takes effect at "
         "the end of the current billing period; access continues until then. Annual "
         "contracts that are cancelled before the anniversary date are not refunded for "
         "the unused months, except within the first 14 days of the initial term."),
        ("Refund eligibility",
         "Monthly subscriptions are refundable only if the request is made within 7 days of "
         "the charge and the account has consumed less than 10 percent of included API "
         "calls. Enterprise contracts may include a refund clause in the order form, which "
         "overrides this policy. Refunds are returned to the original payment method within "
         "5 to 10 business days."),
        ("Service credits are not refunds",
         "Uptime credits granted under the SLA are applied to future invoices and cannot be "
         "converted to cash or refunded. A credit balance expires 12 months after issue if "
         "unused."),
    ]),
    ("sla", "Service Level Agreement", [
        ("Uptime commitment",
         "Northwind Cloud commits to 99.9 percent monthly availability for the API and web "
         "application on Business and Enterprise plans, and 99.5 percent on Starter and "
         "Team. Availability is measured per calendar month from the status page, excluding "
         "scheduled maintenance announced at least 72 hours in advance."),
        ("What counts as downtime",
         "Downtime means that more than 5 percent of requests across all regions return a "
         "5xx status or take longer than 30 seconds. Errors caused by the customer, by "
         "expired credentials, or by requests exceeding documented rate limits are excluded "
         "from the calculation. Third-party integrations are not covered by the SLA."),
    ]),
    ("uptime-credits", "Uptime Credit Claims", [
        ("Credit schedule",
         "If monthly availability falls below the commitment, the customer receives a "
         "credit against the next invoice: 10 percent of the monthly fee below 99.9 percent "
         "and at or above 99.0 percent, 25 percent below 99.0 percent and at or above 98.0 "
         "percent, and 50 percent below 98.0 percent. Credits are the sole remedy for "
         "missed availability."),
        ("How to claim",
         "Claims must be submitted through the support portal within 30 days of the end of "
         "the affected month, using the category Billing, then SLA credit. A claim must "
         "include the workspace ID and the observed error rate. Claims submitted after the "
         "30 day window are rejected automatically. Approved credits appear on the invoice "
         "issued in the following month."),
    ]),
    ("data-retention", "Data Retention and Deletion", [
        ("Retention after cancellation",
         "When a workspace is cancelled, the contents remain readable in a suspended state "
         "for 30 days so that the customer can export. After the 30 day grace period the "
         "workspace is queued for deletion and data is purged from primary storage within "
         "7 days. Encrypted backups containing the data expire within 35 days after purge."),
        ("Early deletion requests",
         "A customer may request immediate deletion instead of waiting for the grace period "
         "by opening a ticket with the category Privacy, then Erasure. Deletion is "
         "irreversible and requires written confirmation from an account owner. Requests "
         "from a non-owner are refused."),
        ("Legal holds",
         "Data covered by a legal hold or a lawful preservation order is exempt from deletion "
         "until the hold is lifted, and the customer is notified unless prohibited by law."),
    ]),
    ("gdpr-dpa", "GDPR and Data Processing Addendum", [
        ("Roles and legal basis",
         "For customer content, Northwind Cloud acts as a processor and the customer is the "
         "controller. For account and billing data Northwind Cloud is an independent "
         "controller. Processing relies on the performance of the contract and, for "
         "marketing email, on consent."),
        ("Signing the DPA",
         "The Data Processing Addendum is available from the trust portal and can be "
         "executed by an account owner in Settings, Legal without a sales conversation. "
         "EU standard contractual clauses are pre-signed and included with the DPA."),
        ("Data location",
         "Customers can choose the EU (Frankfurt) or US (Virginia) region at workspace "
         "creation. The region cannot be changed after the first document is uploaded; "
         "migrating regions requires a new workspace and a support-assisted export."),
    ]),
    ("security-overview", "Security Overview", [
        ("Encryption and isolation",
         "All traffic is encrypted with TLS 1.3 and all data at rest is encrypted with "
         "AES-256. Each workspace is logically isolated; storage keys are per tenant. "
         "Production access by Northwind staff requires hardware-backed multi-factor "
         "authentication and is recorded in an immutable access log."),
        ("Certifications",
         "Northwind Cloud holds SOC 2 Type II, ISO 27001 and ISO 27701. The most recent SOC "
         "2 report can be requested from the trust portal under NDA. Penetration tests are "
         "performed twice a year by an independent firm; the executive summary is shared "
         "with Enterprise customers on request."),
        ("Vulnerability handling",
         "Critical vulnerabilities are remediated within 72 hours of confirmation, high "
         "severity within 14 days, and medium severity within 90 days. Customers can report "
         "issues to security@northwindcloud.example and receive an acknowledgement within "
         "one business day."),
    ]),
    ("sso-scim", "SSO and SCIM Provisioning", [
        ("Single sign-on",
         "SAML 2.0 and OIDC single sign-on are available on the Business and Enterprise "
         "plans. Supported identity providers include Okta, Entra ID, Google Workspace, "
         "OneLogin and JumpCloud. Starter and Team do not include SSO; the upgrade path is "
         "a plan change, not an add-on."),
        ("Provisioning",
         "SCIM 2.0 user and group provisioning is included with Enterprise only. SCIM "
         "de-provisioning disables the account within 60 seconds and removes active "
         "sessions. Group-to-role mapping is configured per workspace and supports up to "
         "50 mappings."),
        ("Enforcement",
         "Once SSO is enforced, password login is disabled for all members, and API keys "
         "issued before enforcement remain valid until manually revoked. A break-glass "
         "administrator account can be enabled on request and is limited to one per "
         "workspace."),
    ]),
    ("api-rate-limits", "API Rate Limits", [
        ("Limits by plan",
         "Rate limits are applied per workspace: Starter 10 requests per second, Team 50, "
         "Business 200, Enterprise negotiated. Bursts of up to three times the sustained "
         "rate are allowed for 10 seconds. Exceeding the limit returns HTTP 429 with a "
         "Retry-After header in seconds."),
        ("Handling 429 responses",
         "Clients should back off exponentially starting at one second and retry up to five "
         "times. Rate limit headers X-RateLimit-Limit, X-RateLimit-Remaining and "
         "X-RateLimit-Reset are returned on every response. Bulk imports should use the "
         "batch endpoint, which accepts up to 500 objects per request."),
        ("Raising a limit",
         "A temporary limit increase for a migration can be requested 5 business days in "
         "advance through the support portal. Permanent increases require a plan change."),
    ]),
    ("webhooks", "Webhooks and Retries", [
        ("Delivery and retries",
         "Webhooks are delivered over HTTPS with a 10 second timeout. Failed deliveries are "
         "retried 8 times with exponential backoff: after 10 seconds, 1 minute, 5 minutes, "
         "30 minutes, 2 hours, 6 hours, 12 hours and 24 hours. After the final attempt the "
         "event is marked failed and can be replayed manually from the dashboard for 30 "
         "days."),
        ("Signature verification",
         "Every request carries an X-Northwind-Signature header containing an HMAC-SHA256 of "
         "the raw body using the endpoint secret. Consumers must verify the signature before "
         "parsing the payload, and must tolerate duplicate events, since delivery is "
         "at-least-once."),
    ]),
    ("integrations", "Supported Integrations", [
        ("Native integrations",
         "Native connectors exist for Salesforce, HubSpot, Slack, Microsoft Teams, Jira, "
         "Zendesk, Snowflake, BigQuery and Google Drive. Connectors sync on a schedule of 15 "
         "minutes at the fastest; real-time sync is available for Salesforce and HubSpot on "
         "Enterprise."),
        ("Custom integrations",
         "Anything not covered natively can be built against the REST API or by subscribing "
         "to webhooks. Northwind does not provide custom connector development, but "
         "maintains a list of certified implementation partners."),
    ]),
    ("onboarding", "Onboarding and Implementation", [
        ("Standard onboarding",
         "A new Business or Enterprise workspace receives a kickoff call within 3 business "
         "days of signature, a data migration workshop in week one, and a go-live review in "
         "week four. The standard implementation takes 30 days for a single region and up "
         "to 90 days for multi-region rollouts."),
        ("Self-serve onboarding",
         "Starter and Team customers use in-product onboarding: a guided import of up to "
         "5,000 documents, a checklist in the admin panel, and access to weekly group "
         "office hours. No dedicated implementation engineer is assigned."),
    ]),
    ("support-plans", "Support Plans and Response Times", [
        ("Response targets",
         "Support is available 24 hours a day for Enterprise and Business, and 8x5 "
         "(09:00 to 18:00 UTC, Monday to Friday) for Starter and Team. First response "
         "targets: P1 one hour, P2 four hours, P3 one business day, P4 three business days. "
         "These are response targets, not resolution targets."),
        ("Severity definitions",
         "P1 is a production outage with no workaround affecting many users. P2 is degraded "
         "functionality with a workaround. P3 is a single-user issue or a question about "
         "configuration. P4 is a feature request or documentation fix. Customers assign the "
         "severity; Northwind may downgrade a severity after review and must explain why in "
         "writing."),
        ("Channels",
         "Tickets can be opened in the support portal, by email to support@northwindcloud.example, "
         "or through the Slack Connect channel included with Enterprise. Phone support is "
         "reserved for P1 incidents on Enterprise."),
    ]),
    ("escalation", "Escalation Matrix", [
        ("Internal escalation",
         "If a P1 ticket is not acknowledged within 30 minutes the on-call engineer is paged. "
         "If it is not resolved or mitigated within 2 hours the incident is escalated to the "
         "Engineering Manager, and after 4 hours to the VP of Engineering. Enterprise "
         "customers may request a written status update every 60 minutes during a P1."),
        ("Customer escalation",
         "A customer who believes a ticket is not progressing may escalate to their account "
         "manager, or to support-escalation@northwindcloud.example. Escalations are reviewed "
         "within one business day and a named owner is assigned."),
    ]),
    ("incident-process", "Incident Response Process", [
        ("Lifecycle",
         "Every incident follows five stages: detection, triage, mitigation, resolution and "
         "review. A public status update is posted within 30 minutes of a confirmed P1 and "
         "every 60 minutes afterwards until resolution. The incident channel is opened in "
         "the status page, not only in email."),
        ("Post-incident review",
         "A written post-incident review is published within 5 business days for any P1 and "
         "any incident that breached the SLA. Enterprise customers receive the review by "
         "email; other customers can request it from support. Reviews include a timeline, "
         "root cause and dated remediation items."),
    ]),
    ("status-page", "Status Page and Maintenance Windows", [
        ("Status page",
         "The status page at status.northwindcloud.example reports component-level health "
         "and incident history. Subscribers receive email or SMS updates. The status page is "
         "hosted outside the production infrastructure so that it remains available during "
         "an outage."),
        ("Maintenance windows",
         "Standard maintenance is scheduled on Sundays between 02:00 and 06:00 UTC and "
         "announced at least 72 hours in advance. Emergency maintenance may be performed "
         "with 2 hours notice. Time inside an announced maintenance window does not count "
         "against the SLA commitment."),
    ]),
    ("backups", "Backup Policy", [
        ("Backup schedule",
         "Primary storage is snapshotted every 6 hours, retained for 35 days. Weekly full "
         "backups are retained for 12 weeks offline. Backups are encrypted with a separate "
         "key from production data. Customers cannot restore snapshots themselves on Starter "
         "and Team; Business and Enterprise can self-restore the last 7 days from the admin "
         "panel."),
        ("Restore requests",
         "A restore request older than 7 days must be opened as a P2 ticket and takes up to "
         "8 business hours. Restores are performed into a new workspace by default so that "
         "the current data is not overwritten."),
    ]),
    ("disaster-recovery", "Disaster Recovery", [
        ("Targets",
         "The recovery point objective is 15 minutes and the recovery time objective is 4 "
         "hours for the API and web application in the primary region. Cross-region failover "
         "is tested twice a year and the results are summarised in the trust portal."),
        ("Customer responsibilities",
         "Customers are responsible for their own integrations, credentials and any data "
         "exported outside Northwind. DR does not cover customer-side automation or "
         "third-party systems that lose connectivity."),
    ]),
    ("data-export", "Data Export and Portability", [
        ("Formats",
         "Workspace data can be exported as JSON Lines or CSV. Documents are exported in "
         "their original format together with a manifest file that records metadata. "
         "Exports are produced asynchronously and a download link is emailed when ready; "
         "the link expires after 72 hours."),
        ("Limits",
         "A single export is capped at 500 GB. Larger workspaces must be exported in "
         "partitions by date range. Export generation usually takes under 2 hours for 100 GB "
         "and is not counted against API rate limits."),
    ]),
    ("subprocessors", "Subprocessors and Subcontractors", [
        ("Current subprocessors",
         "Northwind Cloud uses Amazon Web Services for compute and storage, Cloudflare for "
         "CDN and DDoS protection, Stripe for payment processing, and SendGrid for "
         "transactional email. An up-to-date list with purpose and location is published on "
         "the trust portal."),
        ("Change notification",
         "Customers are notified at least 30 days before a new subprocessor is engaged and "
         "may object in writing. If an objection cannot be resolved, the customer may "
         "terminate the affected contract without penalty."),
    ]),
    ("audit-logs", "Audit Logs", [
        ("Coverage",
         "Audit logs record authentication events, permission changes, document downloads, "
         "API key creation and deletion, and configuration changes. Logs are retained for "
         "400 days on Business and Enterprise and 90 days on Starter and Team."),
        ("Access and export",
         "Administrators can view logs in the admin panel and stream them to an external "
         "SIEM through the audit log export endpoint. Exported records are immutable; "
         "deleting a user does not delete their historical entries."),
    ]),
    ("seats", "Seats and License Management", [
        ("Adding and removing seats",
         "Seats are billed per active member. Adding a seat mid-cycle prorates the charge to "
         "the end of the month. Removing a seat does not generate a credit; the seat becomes "
         "available for reuse until the period ends."),
        ("Viewer seats",
         "Viewer seats are read-only and cost 20 percent of a full seat on Team, Business and "
         "Enterprise. Starter has no viewer seats. Service accounts used for API access are "
         "free and do not count toward the seat limit."),
    ]),
    ("free-trial", "Free Trial Terms", [
        ("Trial length",
         "Every new workspace starts with a 14 day trial of the Business plan, including all "
         "features except SCIM provisioning and custom data residency. No credit card is "
         "required to start. At the end of the trial the workspace downgrades to a read-only "
         "state for 14 days before data is scheduled for deletion."),
        ("Extensions",
         "One 14 day extension is available on request through support. A second extension "
         "requires approval from an account manager and is granted only for active "
         "evaluation projects."),
    ]),
    ("discounts", "Discounts and Nonprofit Pricing", [
        ("Volume and term discounts",
         "Discounts are available for annual prepayment (10 percent), three-year commitments "
         "(20 percent) and volumes above 250 seats (negotiated). Discounts do not stack; the "
         "single largest applicable discount applies."),
        ("Nonprofit and education",
         "Registered nonprofits and accredited educational institutions receive 50 percent "
         "off the Team and Business plans after verification, which typically takes 5 "
         "business days. The discount applies to subscription fees, not to overage."),
    ]),
]


DOCS_EXTRA = [
    ("api-authentication", "API Authentication", [
        ("Keys and scopes",
         "Requests authenticate with a bearer API key. Each key is bound to a set of scopes: "
         "read, write, admin and billing. A key never inherits more scope than the member "
         "who created it. Keys are shown once at creation and cannot be retrieved later; "
         "only the last four characters are displayed afterwards."),
        ("Rotation",
         "Keys can be rotated without downtime by creating a second key, deploying it, then "
         "revoking the first. Northwind recommends rotation every 90 days. Revocation takes "
         "effect within 30 seconds across all regions. Keys that appear in a public "
         "repository are revoked automatically when detected."),
    ]),
    ("api-pagination", "Pagination and Filtering", [
        ("Cursors",
         "List endpoints return up to 100 objects per page and use cursor pagination, not "
         "offsets, because offsets drift when records are inserted concurrently. The response "
         "contains a next_cursor field; pass it as the cursor parameter to fetch the next "
         "page. A null cursor means the end of the result set."),
        ("Filtering",
         "Every list endpoint accepts created_after, created_before and updated_after as "
         "ISO 8601 timestamps. Filters combine with AND. Sorting is limited to created_at "
         "and updated_at; arbitrary sort fields are not supported."),
    ]),
    ("api-errors", "API Error Codes", [
        ("Status codes",
         "400 means a malformed request or failed validation and includes a field-level "
         "error list. 401 means a missing or invalid key. 403 means the key lacks the "
         "required scope. 404 means the object does not exist or is not visible to the key. "
         "409 means a conflict, most often a duplicate idempotency key. 422 means the payload "
         "parsed but violated a business rule. 429 is rate limiting. 5xx are server faults "
         "and are always safe to retry with backoff."),
        ("Error body",
         "Errors return a JSON body with code, message and request_id. Support requests "
         "without a request_id are slower to investigate, so log it on every failure."),
    ]),
    ("api-idempotency", "Idempotent Writes", [
        ("Idempotency keys",
         "POST requests accept an Idempotency-Key header of up to 255 characters. A repeated "
         "request with the same key returns the original response instead of creating a "
         "second object. Keys are retained for 24 hours; reusing a key with a different "
         "payload returns 409."),
    ]),
    ("webhook-events", "Webhook Event Catalogue", [
        ("Event types",
         "Available events are document.created, document.updated, document.deleted, "
         "workspace.member_added, workspace.member_removed, export.completed, "
         "billing.invoice_paid and billing.payment_failed. Each endpoint can subscribe to a "
         "subset of events; subscribing to all events is allowed but discouraged because it "
         "increases delivery volume."),
        ("Payload envelope",
         "Every payload contains id, type, created_at and data. The id is stable across "
         "retries of the same event. Ordering is not guaranteed, so consumers should "
         "compare created_at rather than assume arrival order."),
    ]),
    ("roles-permissions", "Roles and Permissions", [
        ("Built-in roles",
         "Owner can transfer ownership and delete the workspace; there is exactly one owner. "
         "Admin manages members, billing and integrations. Editor creates and modifies "
         "documents. Viewer reads documents and cannot export. Billing is a separate role "
         "that sees invoices but no content."),
        ("Custom roles",
         "Enterprise workspaces can define custom roles from a permission list of 34 "
         "individual permissions. Custom roles cannot grant permissions the workspace plan "
         "does not include."),
    ]),
    ("search-syntax", "Search Syntax", [
        ("Operators",
         "The in-product search supports quoted phrases for exact matching, minus to exclude "
         "a term, and tag: to filter by tag. Operators combine with implicit AND. Wildcards "
         "are not supported."),
        ("Ranking",
         "Results rank by a combination of term frequency, recency and document popularity "
         "within the workspace. Recency contributes up to 20 percent of the score and can be "
         "disabled per user in settings."),
    ]),
    ("data-model", "Core Data Model", [
        ("Objects",
         "A workspace contains collections; a collection contains documents; a document has "
         "revisions and tags. Tags are workspace-scoped, not global. Deleting a collection "
         "moves its documents to the workspace root rather than deleting them."),
        ("Identifiers",
         "All objects use prefixed identifiers: ws_ for workspaces, doc_ for documents, "
         "col_ for collections. Identifiers are stable and never reused after deletion."),
    ]),
    ("browsers", "Supported Browsers", [
        ("Support matrix",
         "The web application supports the last two major versions of Chrome, Edge, Firefox "
         "and Safari. Internet Explorer is not supported. The application requires "
         "JavaScript and WebSocket support; environments that block WebSockets fall back to "
         "polling every 30 seconds."),
    ]),
    ("accessibility", "Accessibility", [
        ("Conformance",
         "The web application targets WCAG 2.1 level AA. A voluntary product accessibility "
         "template is available on request. Known gaps are documented in the trust portal, "
         "currently limited to the diagram editor, which lacks full keyboard navigation."),
    ]),
    ("localization", "Interface Languages", [
        ("Available languages",
         "The interface is available in English, German, French, Spanish, Portuguese "
         "(Brazil), Japanese and Simplified Chinese. Documents may be written in any "
         "language, but retrieval quality is tuned and evaluated for English content."),
    ]),
    ("professional-services", "Professional Services", [
        ("Scope and rates",
         "Implementation, migration scripting and custom reporting are delivered as "
         "professional services at 180 USD per hour, sold in blocks of 20 hours. Blocks "
         "expire 12 months after purchase. Services are not covered by the subscription."),
        ("When they are required",
         "Multi-region rollouts and migrations above 2 TB usually require a services "
         "engagement. Northwind will state the requirement in writing before signature."),
    ]),
    ("training", "Training and Certification", [
        ("Northwind Academy",
         "Self-paced courses are free for all customers. Administrator and Developer "
         "certification exams cost 150 USD each; certified individuals keep the credential "
         "for two years. Live instructor sessions are included with Enterprise onboarding."),
    ]),
    ("partner-program", "Partner and Reseller Program", [
        ("Referral programme",
         "Referral partners receive 15 percent of the first year's subscription value, paid "
         "60 days after the customer's first invoice is settled. Referrals are attributed to "
         "the partner who registered the opportunity first."),
        ("Resellers",
         "Authorised resellers buy at a margin of 20 to 30 percent depending on volume and "
         "handle first-line support for their customers. Resellers cannot set prices below "
         "the published list price without written approval."),
    ]),
    ("cookies", "Cookies and Tracking", [
        ("Categories",
         "Strictly necessary cookies keep sessions and cannot be disabled. Preference "
         "cookies remember language and layout. Analytics cookies are optional and off until "
         "consent is given. The marketing site uses no third-party advertising cookies."),
    ]),
    ("acceptable-use", "Acceptable Use Policy", [
        ("Prohibited content",
         "Customers may not store malware, run denial-of-service tests without written "
         "permission, or use the service to process data they are not lawfully entitled to "
         "process. Bulk scraping of third-party sites through the service is prohibited."),
        ("Enforcement",
         "Suspected violations are reviewed by the trust and safety team, and the customer is "
         "notified within 2 business days unless notification would create legal risk. "
         "Repeated violations may lead to suspension of write access."),
    ]),
    ("terms-summary", "Terms of Service Summary", [
        ("Key points",
         "The subscription is non-exclusive and non-transferable. The customer owns its "
         "content and grants Northwind a licence to host and process it. Liability is capped "
         "at the fees paid in the 12 months before the claim, except for data breaches caused "
         "by gross negligence, where the cap does not apply."),
        ("Governing law",
         "Contracts are governed by the laws of the Republic of Ireland, and disputes are "
         "heard in Dublin, unless the order form states otherwise."),
    ]),
    ("api-versioning", "API Versioning and Deprecation", [
        ("Versioning",
         "The API is versioned in the path, for example /v2/documents. A new major version is "
         "released at most once a year. Backwards-incompatible changes never ship inside an "
         "existing version."),
        ("Deprecation",
         "Deprecated endpoints keep working for at least 12 months after the announcement and "
         "return a Deprecation header with the sunset date. Customers are notified by email "
         "and through the changelog at least 90 days before the sunset date."),
    ]),
    ("changelog", "Release Cadence", [
        ("Shipping rhythm",
         "The web application deploys several times per week behind feature flags. The API "
         "adds non-breaking fields continuously. A public changelog entry is published for "
         "every customer-visible change."),
    ]),
    ("uptime-history", "Historical Availability", [
        ("Published history",
         "Monthly availability for the last 24 months is published on the trust portal, "
         "broken down by region and by component. The trailing 12 month average has been "
         "99.96 percent for the API in the EU region and 99.92 percent in the US region."),
    ]),
    ("enterprise-quote", "Enterprise Quoting Process", [
        ("Steps",
         "Enterprise pricing is quoted after a scoping call and a security review. Quotes are "
         "valid for 30 days. The process from scoping call to countersigned order form "
         "typically takes 3 to 6 weeks; security review is the most common source of delay."),
        ("Custom terms",
         "Enterprise contracts may override uptime commitment, payment terms, liability caps "
         "and data residency. Any clause in the order form overrides the public policy page "
         "for that customer."),
    ]),
    ("security-questionnaire", "Security Questionnaires", [
        ("Standard responses",
         "Completed CAIQ and SIG questionnaires are available from the trust portal without a "
         "sales conversation. Custom questionnaires are answered within 10 business days; "
         "longer questionnaires are queued behind existing requests."),
    ]),
]

DOCS += DOCS_EXTRA


def main() -> None:
    CORPUS.mkdir(parents=True, exist_ok=True)
    for name, title, sections in DOCS:
        parts = ["# %s" % title, ""]
        for heading, body in sections:
            parts += ["## %s" % heading, "", body, ""]
        (CORPUS / ("%s.md" % name)).write_text("\n".join(parts), encoding="utf-8")

    golden = [
        # (question, [gold doc ids], note)
        ("What is the API call allowance on the Team plan?", ["pricing-plans"], "identifier"),
        ("How much does an extra 1,000 API calls cost on Business?", ["pricing-plans"], "number"),
        ("Do I get throttled if I go over my included usage?", ["pricing-plans"], "paraphrase"),
        ("When does my subscription renew each month?", ["billing-cycle"], "date"),
        ("What happens if an invoice is 50 days late?", ["billing-cycle"], "threshold"),
        ("Is there a fee for paying by wire transfer?", ["billing-cycle"], "number"),
        ("Can I get my money back after cancelling an annual plan?", ["refunds"], "paraphrase"),
        ("How long does a refund take to arrive?", ["refunds"], "number"),
        ("Can SLA credits be paid out in cash?", ["refunds", "uptime-credits"], "cross-doc"),
        ("What uptime do you promise on the Starter plan?", ["sla"], "identifier"),
        ("Does scheduled maintenance count against the SLA?", ["sla", "status-page"], "cross-doc"),
        ("How much credit do I get if availability drops to 98.5 percent?", ["uptime-credits"], "number"),
        ("How long do I have to file an SLA claim?", ["uptime-credits"], "number"),
        ("After I cancel, how long before my documents are gone for good?", ["data-retention"], "paraphrase"),
        ("Can I make you delete my workspace immediately instead of waiting?", ["data-retention"], "paraphrase"),
        ("Who can request erasure of a workspace?", ["data-retention"], "detail"),
        ("Do you act as a processor or a controller for our content?", ["gdpr-dpa"], "detail"),
        ("Can I sign the DPA without talking to sales?", ["gdpr-dpa"], "paraphrase"),
        ("Can I move my workspace to another region later?", ["gdpr-dpa"], "paraphrase"),
        ("What encryption is used for data at rest?", ["security-overview"], "identifier"),
        ("Which compliance certificates do you hold?", ["security-overview"], "identifier"),
        ("How fast are critical vulnerabilities fixed?", ["security-overview"], "number"),
        ("Is SSO included in the Team plan?", ["sso-scim"], "detail"),
        ("Which identity providers are supported?", ["sso-scim"], "list"),
        ("What happens to API keys when we enforce SSO?", ["sso-scim"], "detail"),
        ("What is the rate limit for the Business plan?", ["api-rate-limits"], "number"),
        ("What should my client do when it receives a 429?", ["api-rate-limits"], "paraphrase"),
        ("How many objects can one batch request contain?", ["api-rate-limits"], "number"),
        ("How many times will a failed webhook be retried?", ["webhooks"], "number"),
        ("How do I verify that a webhook really came from you?", ["webhooks"], "paraphrase"),
        ("Can webhook deliveries arrive more than once?", ["webhooks"], "detail"),
        ("How long does a standard implementation take?", ["onboarding"], "number"),
        ("Do Starter customers get a dedicated implementation engineer?", ["onboarding"], "detail"),
        ("What is the first response target for a P1?", ["support-plans"], "number"),
        ("How long is support available on the Team plan?", ["support-plans"], "paraphrase"),
        ("Who decides the severity of my ticket?", ["support-plans"], "detail"),
        ("How often are status updates posted during a major outage?", ["incident-process"], "number"),
        ("When do I get a post-incident review?", ["incident-process"], "number"),
        ("When is routine maintenance scheduled?", ["status-page"], "paraphrase"),
        ("How often are backups taken and how long are they kept?", ["backups"], "number"),
        ("Can I restore yesterday's data myself?", ["backups"], "paraphrase"),
        ("What is the recovery time objective?", ["disaster-recovery"], "identifier"),
        ("What export formats are available?", ["data-export"], "list"),
        ("How long does an export download link stay valid?", ["data-export"], "number"),
        ("Which companies process my data on your behalf?", ["subprocessors"], "paraphrase"),
        ("How much notice do you give before adding a new subprocessor?", ["subprocessors"], "number"),
        ("How long are audit logs kept on Business?", ["audit-logs"], "number"),
        ("Do service accounts consume seats?", ["seats"], "detail"),
        ("How long is the free trial and is a card required?", ["free-trial"], "paraphrase"),
        ("Do you offer discounts to nonprofits?", ["discounts"], "number"),
        ("How do I rotate an API key without downtime?", ["api-authentication"], "paraphrase"),
        ("What happens if my API key leaks into a public repo?", ["api-authentication"], "detail"),
        ("Why does the API use cursors instead of page numbers?", ["api-pagination"], "paraphrase"),
        ("How many objects come back in one page?", ["api-pagination"], "number"),
        ("What does a 409 response mean?", ["api-errors"], "identifier"),
        ("Is it safe to retry a 5xx error?", ["api-errors"], "detail"),
        ("How do I stop a retried POST from creating duplicates?", ["api-idempotency"], "paraphrase"),
        ("How long is an idempotency key remembered?", ["api-idempotency"], "number"),
        ("Which webhook events can I subscribe to?", ["webhook-events"], "list"),
        ("Are webhook events delivered in order?", ["webhook-events"], "detail"),
        ("Who can delete the workspace?", ["roles-permissions"], "detail"),
        ("Can I build my own role with specific permissions?", ["roles-permissions"], "paraphrase"),
        ("How do I search for an exact phrase?", ["search-syntax"], "paraphrase"),
        ("What is the difference between a collection and a tag?", ["data-model"], "detail"),
        ("Which browsers are supported?", ["browsers"], "list"),
        ("Does the app meet WCAG AA?", ["accessibility"], "identifier"),
        ("Which languages is the interface translated into?", ["localization"], "list"),
        ("What does professional services cost per hour?", ["professional-services"], "number"),
        ("How much does a certification exam cost?", ["training"], "number"),
        ("What commission do referral partners earn?", ["partner-program"], "number"),
        ("Can resellers discount below list price?", ["partner-program"], "detail"),
        ("Are analytics cookies on by default?", ["cookies"], "detail"),
        ("Is scraping third party sites through your product allowed?", ["acceptable-use"], "paraphrase"),
        ("What is the liability cap in the contract?", ["terms-summary"], "detail"),
        ("Which law governs the contract?", ["terms-summary"], "detail"),
        ("How long do deprecated endpoints keep working?", ["api-versioning"], "number"),
        ("How often is the changelog updated?", ["changelog"], "paraphrase"),
        ("What was the average uptime over the last year?", ["uptime-history"], "number"),
        ("How long does an enterprise deal take to close?", ["enterprise-quote"], "number"),
        ("Can an order form override the public policy page?", ["enterprise-quote"], "paraphrase"),
        ("Do you answer custom security questionnaires?", ["security-questionnaire"], "paraphrase"),
    ]

    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    with GOLDEN.open("w", encoding="utf-8") as fh:
        for question, gold, note in golden:
            fh.write(json.dumps({"question": question, "gold": gold, "type": note},
                                ensure_ascii=False) + "\n")

    print("wrote %d documents to %s" % (len(DOCS), CORPUS))
    print("wrote %d questions to %s" % (len(golden), GOLDEN))


if __name__ == "__main__":
    main()
