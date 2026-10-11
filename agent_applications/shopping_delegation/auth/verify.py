"""Verify the user credential carried in an NLIP authorization token submessage."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jwt

from nlip_sdk import NLIPMessage


class AuthenticationError(Exception):
    """Raised when a message carries no valid user credential."""


@dataclass(frozen=True, slots=True)
class Principal:
    sub: str
    email: str
    name: str


class IdentityVerifier:
    """Check an ID token's signature, issuer, audience, and lifetime."""

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        public_key_loader: Callable[[], bytes],
    ) -> None:
        self._issuer = issuer
        self._audience = audience
        self._public_key_loader = public_key_loader
        self._public_key: bytes | None = None

    def verify(self, token: str) -> Principal:
        # Load lazily: the mock key exists only after the first user login.
        if self._public_key is None:
            try:
                self._public_key = self._public_key_loader()
            except FileNotFoundError as error:
                raise AuthenticationError(str(error)) from error
        try:
            claims = jwt.decode(
                token,
                self._public_key,
                algorithms=["RS256"],
                audience=self._audience,
                issuer=self._issuer,
                options={"require": ["exp", "iat", "iss", "aud", "sub"]},
            )
        except jwt.PyJWTError as error:
            raise AuthenticationError(f"invalid credential: {error}") from error
        return Principal(
            sub=claims["sub"],
            email=claims.get("email", ""),
            name=claims.get("name", claims["sub"]),
        )


class SessionAuthenticator:
    """Authenticate every message and bind each NLIP session to one user.

    Binding stops a stolen or replayed conversation token from resuming another
    user's session, because the framework resolves sessions from that token alone.
    """

    def __init__(self, verifier: IdentityVerifier) -> None:
        self.verifier = verifier
        self._bindings: dict[str, str] = {}

    def authenticate(self, message: NLIPMessage, session_id: str) -> tuple[Principal, str]:
        token = message.extract_authentication_token()
        if token is None:
            raise AuthenticationError("no authorization token in the message")
        principal = self.verifier.verify(token)
        bound = self._bindings.setdefault(session_id, principal.sub)
        if bound != principal.sub:
            raise AuthenticationError("this conversation belongs to a different user")
        return principal, token

    def forget(self, session_id: str) -> None:
        self._bindings.pop(session_id, None)
