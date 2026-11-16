# Security policy

Local LLM runs code, fetches web pages, and connects to your accounts through MCP servers, so security reports are
taken seriously.

## Reporting a vulnerability

Please report vulnerabilities privately through
[GitHub's private vulnerability reporting](https://github.com/maxmoundas/local-llm/security/advisories/new)
rather than in a public issue. Include steps to reproduce and the impact you expect. You should get a response
within a week.

## Scope

Examples of issues that are in scope:

- Reaching the backend from another device or web page without the `ACCESS_TOKEN` (including DNS rebinding and CSRF)
- Escaping the Docker code sandbox, or leaking host environment variables or API keys into it
- Prompt injection that makes the agent send data out without the approval prompt
- Preview iframes that gain access to the app or its API
- Path traversal through attachments, backups/restore, or agent workspace files

Running the "Local" (unsandboxed) code mode, or exposing the server with no access token, is a documented trade-off
and not a vulnerability by itself.
