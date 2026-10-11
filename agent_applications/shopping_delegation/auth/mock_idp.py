"""Offline stand-in for an OIDC identity provider such as Google.

It issues RS256-signed ID tokens with the same core claims Google uses (iss, aud,
sub, email, email_verified, name, iat, exp), so the agents verify mock and real
logins the same way. The signing key is generated on this machine at first login
and is git-ignored, so no secret is ever committed.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

PRIVATE_KEY_FILE = "mock_idp_private.pem"
PUBLIC_KEY_FILE = "mock_idp_public.pem"


def ensure_keys(key_dir: Path) -> None:
    """Create the signing key pair once; later calls reuse it."""
    private_path = key_dir / PRIVATE_KEY_FILE
    public_path = key_dir / PUBLIC_KEY_FILE
    if private_path.exists() and public_path.exists():
        return
    key_dir.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )


def read_public_key(key_dir: Path) -> bytes:
    path = key_dir / PUBLIC_KEY_FILE
    if not path.exists():
        raise FileNotFoundError(
            "Mock identity provider key not found. Log in with the user client first."
        )
    return path.read_bytes()


class MockIdentityProvider:
    """Issue ID tokens for the test users defined in the accounts file."""

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        key_dir: Path,
        users: Mapping[str, Mapping[str, Any]],
        ttl_seconds: int = 3600,
    ) -> None:
        self._issuer = issuer
        self._audience = audience
        self._key_dir = Path(key_dir)
        self._users = {name.casefold(): user for name, user in users.items()}
        self._ttl_seconds = ttl_seconds

    @property
    def usernames(self) -> tuple[str, ...]:
        return tuple(self._users)

    def login(self, username: str) -> str:
        user = self._users.get(username.casefold())
        if user is None:
            choices = ", ".join(self.usernames)
            raise ValueError(f"Unknown mock user {username!r}. Choose from: {choices}.")
        ensure_keys(self._key_dir)
        now = int(time.time())
        claims = {
            "iss": self._issuer,
            "aud": self._audience,
            "sub": user["sub"],
            "email": user["email"],
            "email_verified": True,
            "name": user["name"],
            "iat": now,
            "exp": now + self._ttl_seconds,
        }
        private_key = (self._key_dir / PRIVATE_KEY_FILE).read_bytes()
        return jwt.encode(claims, private_key, algorithm="RS256")
