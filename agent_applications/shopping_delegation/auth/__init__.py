"""User login and credential verification for the shopping-delegation app."""

from __future__ import annotations

import json
from typing import Any

from ..settings import resolve_path
from .mock_idp import MockIdentityProvider, read_public_key
from .verify import AuthenticationError, IdentityVerifier, Principal, SessionAuthenticator

__all__ = [
    "AuthenticationError",
    "IdentityVerifier",
    "MockIdentityProvider",
    "Principal",
    "SessionAuthenticator",
    "build_authenticator",
    "build_identity_provider",
]


def _auth_settings(config: dict[str, Any]) -> dict[str, Any]:
    auth = config["auth"]
    if auth.get("mode") != "mock":
        raise ValueError(f"Unsupported auth mode {auth.get('mode')!r}; only 'mock' exists.")
    return auth


def build_identity_provider(config: dict[str, Any]) -> MockIdentityProvider:
    auth = _auth_settings(config)
    accounts_path = resolve_path(config["data"]["accounts"])
    users = json.loads(accounts_path.read_text(encoding="utf-8"))
    return MockIdentityProvider(
        issuer=auth["issuer"],
        audience=auth["audience"],
        key_dir=resolve_path(auth["key_dir"]),
        users=users,
        ttl_seconds=int(auth.get("token_ttl_seconds", 3600)),
    )


def build_authenticator(config: dict[str, Any]) -> SessionAuthenticator:
    auth = _auth_settings(config)
    key_dir = resolve_path(auth["key_dir"])
    verifier = IdentityVerifier(
        issuer=auth["issuer"],
        audience=auth["audience"],
        public_key_loader=lambda: read_public_key(key_dir),
    )
    return SessionAuthenticator(verifier)
