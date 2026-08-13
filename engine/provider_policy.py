"""Credential-bound provider URL validation.

Provider endpoints are deployment configuration, but a typo or stale variable
must not redirect a production credential to an unrelated host.  Custom
OpenAI-compatible gateways remain possible only through an explicit opt-in.
"""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlsplit
import re


ProviderName = Literal["runpod", "zai", "research_proxy"]

_ALLOWED_HOSTS: dict[ProviderName, tuple[str, ...]] = {
    "runpod": ("api.runpod.ai", "api.runpod.io", "rest.runpod.io"),
    "zai": ("open.bigmodel.cn",),
    "research_proxy": ("ai.hackclub.com",),
}


def validate_provider_url(
    url: str,
    provider: ProviderName,
    *,
    allow_custom: bool = False,
) -> str:
    """Return a normalized HTTPS base URL or fail before attaching credentials."""
    candidate = url.strip().rstrip("/")
    parsed = urlsplit(candidate)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError(f"{provider} endpoint must be an absolute HTTPS URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(f"{provider} endpoint cannot contain credentials, query, or fragment")

    hostname = parsed.hostname.lower().rstrip(".")
    allowed = _ALLOWED_HOSTS[provider]
    trusted = hostname in allowed or (
        provider == "runpod"
        and any(hostname.endswith(f".{root}") for root in allowed)
    )
    if not trusted and not allow_custom:
        raise ValueError(
            f"refusing to send {provider} credentials to untrusted host {hostname!r}; "
            "set HYBRIDMIND_ALLOW_CUSTOM_PROVIDER_URLS=true only for an audited gateway"
        )
    return candidate


def validate_runpod_endpoint_id(endpoint_id: str) -> str:
    """Reject path/query injection in RunPod endpoint identifiers."""
    candidate = endpoint_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", candidate):
        raise ValueError("RunPod endpoint ID contains unsupported characters")
    return candidate
