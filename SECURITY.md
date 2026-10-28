# Security Policy

## Reporting a vulnerability

**Do not open a public issue.** Report privately via
[GitHub Security Advisories](https://github.com/arnabwithab/ai-therapist/security/advisories/new)
(private vulnerability reporting).

Include: what you found, where (file/endpoint/commit), steps to reproduce,
and impact if exploited. Redact secrets — never paste tokens or keys.

Expect an initial response within 7 days. This is a college demo project
maintained part-time, so fixes land on a best-effort basis; critical issues
(RCE, secret exposure, prompt/data exfiltration via the API) come first.

## Scope

In scope: `backend/`, `src/`, `Makefile` targets, and the default configs
that ship in this repo. Out of scope: the `frontend/anse` submodule
(report upstream), third-party model weights, and findings that require a
committed `.env` or other ignored file (see "Never commit" in
`CONTRIBUTING.md` — secrets must never be in the repo in the first place).

## Ground rules

- No secrets in code, issues, or PRs (`.env`, `HF_TOKEN`, API keys).
- Dependencies are pinned in `uv.lock`; flag vulnerable pins via a
  private report, not a public PR, if exploitation is non-obvious.
