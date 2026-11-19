# SSO and SCIM Provisioning

## Single sign-on

SAML 2.0 and OIDC single sign-on are available on the Business and Enterprise plans. Supported identity providers include Okta, Entra ID, Google Workspace, OneLogin and JumpCloud. Starter and Team do not include SSO; the upgrade path is a plan change, not an add-on.

## Provisioning

SCIM 2.0 user and group provisioning is included with Enterprise only. SCIM de-provisioning disables the account within 60 seconds and removes active sessions. Group-to-role mapping is configured per workspace and supports up to 50 mappings.

## Enforcement

Once SSO is enforced, password login is disabled for all members, and API keys issued before enforcement remain valid until manually revoked. A break-glass administrator account can be enabled on request and is limited to one per workspace.
