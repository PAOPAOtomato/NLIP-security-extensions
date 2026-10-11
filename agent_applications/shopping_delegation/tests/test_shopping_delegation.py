"""Shopping-delegation app tests. Scripted stand-ins replace the LLMs, so no API calls."""

import asyncio
import base64
import json
from collections.abc import Sequence
from typing import Any

import httpx
import pytest

from agent_applications.shopping_delegation.auth import (
    AuthenticationError,
    IdentityVerifier,
    MockIdentityProvider,
    build_authenticator,
    build_identity_provider,
)
from agent_applications.shopping_delegation.auth.mock_idp import read_public_key
from agent_applications.shopping_delegation.catalog import Catalog
from agent_applications.shopping_delegation.payment_agent import PaymentAgent
from agent_applications.shopping_delegation.protocol import extract_payment_request
from agent_applications.shopping_delegation.settings import APP_DIR
from agent_applications.shopping_delegation.shopping_agent import ShoppingAgent
from agent_applications.shopping_delegation.store import PaymentStore, StoreError
from nlip_sdk import NLIPMessage

ALICE_SUB = "mock-alice-001"


def make_config(tmp_path) -> dict[str, Any]:
    return {
        "entities": {
            "shopping": {"endpoint": "http://shopping.test/nlip", "model": "unused"},
            "payment": {"endpoint": "http://payment.test/nlip", "model": "unused"},
        },
        "auth": {
            "mode": "mock",
            "issuer": "https://mock-idp.test",
            "audience": "shopping-delegation",
            "key_dir": str(tmp_path / "keys"),
            "token_ttl_seconds": 600,
        },
        "data": {
            "catalog": str(APP_DIR / "data" / "catalog.json"),
            "accounts": str(APP_DIR / "data" / "accounts.json"),
            "ledger": str(tmp_path / "ledger.jsonl"),
        },
    }


def make_store(config: dict[str, Any]) -> PaymentStore:
    data = config["data"]
    return PaymentStore.load(data["accounts"], Catalog.load(data["catalog"]), data["ledger"])


class ScriptedAgent:
    """Stands in for the LLM: calls one tool with fixed arguments, then replies."""

    def __init__(self, tools: Sequence[Any], tool_name: str | None, arguments: dict) -> None:
        self.tools = {tool.name: tool for tool in tools}
        self.tool_name = tool_name
        self.arguments = arguments
        self.prompts: list[str] = []

    async def chat(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.tool_name is None:
            return "ok"
        result = await self.tools[self.tool_name].ainvoke(self.arguments)
        return f"{self.tool_name} -> {result}"


def scripted(tool_name: str | None, arguments: dict, created: list[ScriptedAgent]):
    def factory(system_prompt: str, tools: Sequence[Any]) -> ScriptedAgent:
        agent = ScriptedAgent(tools, tool_name, arguments)
        created.append(agent)
        return agent

    return factory


def connect(shopping: ShoppingAgent, payment: PaymentAgent, received: list[NLIPMessage]):
    """Route the shopping agent's outbound NLIP requests to the payment agent in-process."""

    async def post(destination: str, message: NLIPMessage) -> httpx.Response:
        wire = NLIPMessage.model_validate(message.to_dict())
        received.append(wire)
        response = await payment.process_exchange(wire)
        return httpx.Response(
            200,
            json=response.to_dict(),
            request=httpx.Request("POST", shopping.nlip_client.endpoints[destination]),
        )

    shopping.nlip_client._post = post  # type: ignore[method-assign]


def user_message(text: str, token: str | None) -> NLIPMessage:
    message = NLIPMessage.text(text)
    if token is not None:
        message.add_authentication_token(token)
    return message


def forge_subject(token: str, sub: str) -> str:
    """Rewrite the token's subject claim while keeping the original signature."""
    header, payload, signature = token.split(".")
    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    claims["sub"] = sub
    forged = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    return f"{header}.{forged}.{signature}"


def test_mock_login_tokens_verify_and_bad_tokens_are_rejected(tmp_path) -> None:
    config = make_config(tmp_path)
    token = build_identity_provider(config).login("alice")
    verifier = build_authenticator(config).verifier
    assert verifier.verify(token).sub == ALICE_SUB

    with pytest.raises(AuthenticationError):
        verifier.verify(forge_subject(token, "mock-bob-002"))

    key_dir = tmp_path / "keys"
    other_audience = IdentityVerifier(
        issuer="https://mock-idp.test",
        audience="another-app",
        public_key_loader=lambda: read_public_key(key_dir),
    )
    with pytest.raises(AuthenticationError):
        other_audience.verify(token)

    expired = MockIdentityProvider(
        issuer="https://mock-idp.test",
        audience="shopping-delegation",
        key_dir=key_dir,
        users={"alice": {"sub": ALICE_SUB, "email": "a@example.com", "name": "A"}},
        ttl_seconds=-10,
    ).login("alice")
    with pytest.raises(AuthenticationError):
        verifier.verify(expired)


def test_store_charges_catalog_prices_and_records_every_action(tmp_path) -> None:
    store = make_store(make_config(tmp_path))
    assert store.catalog.search("usb-c charger")[0]["category"] == "chargers"

    order = store.place_order(ALICE_SUB, "p-1002", 2, session_id="s1")
    assert order["total"] == 79.98
    store.send_gift_card(ALICE_SUB, "friend@example.com", 25, session_id="s1")
    with pytest.raises(StoreError):
        store.place_order(ALICE_SUB, "p-9999")
    with pytest.raises(StoreError):
        store.send_gift_card(ALICE_SUB, "friend@example.com", 10_000)

    ledger = store.ledger()
    assert [entry["action"] for entry in ledger] == ["place_order", "send_gift_card"]
    assert {entry["principal"] for entry in ledger} == {ALICE_SUB}


def test_purchase_travels_from_user_through_shopping_agent_to_payment_agent(tmp_path) -> None:
    config = make_config(tmp_path)
    token = build_identity_provider(config).login("alice")
    shopping_brains: list[ScriptedAgent] = []
    payment_brains: list[ScriptedAgent] = []
    store = make_store(config)
    shopping = ShoppingAgent(
        config,
        authenticator=build_authenticator(config),
        catalog=Catalog.load(config["data"]["catalog"]),
        agent_factory=scripted(
            "request_payment",
            {"action": "place_order", "product_id": "p-1001", "quantity": 1, "note": "Buy it."},
            shopping_brains,
        ),
    )
    payment = PaymentAgent(
        config,
        authenticator=build_authenticator(config),
        store=store,
        agent_factory=scripted("place_order", {"product_id": "p-1001", "quantity": 1}, payment_brains),
    )
    received: list[NLIPMessage] = []
    connect(shopping, payment, received)

    response = asyncio.run(
        shopping.process_exchange(user_message("Buy me the VoltMax charger.", token))
    )

    # The payment agent got a structured request plus the user's forwarded credential.
    [delegated] = received
    assert extract_payment_request(delegated) == {
        "action": "place_order",
        "product_id": "p-1001",
        "quantity": 1,
    }
    assert delegated.extract_authentication_token() == token
    assert '"product_id": "p-1001"' in payment_brains[0].prompts[0]
    # It acted as Alice, charged the catalog price, and the reply reached the user.
    [entry] = store.ledger()
    assert entry["principal"] == ALICE_SUB
    assert entry["details"]["total"] == 24.99
    assert entry["details"]["order_id"] in (response.extract_text(language=None) or "")


def test_agents_reject_missing_forged_or_switched_credentials(tmp_path) -> None:
    config = make_config(tmp_path)
    identity_provider = build_identity_provider(config)
    alice = identity_provider.login("alice")
    bob = identity_provider.login("bob")
    brains: list[ScriptedAgent] = []
    shopping = ShoppingAgent(
        config,
        authenticator=build_authenticator(config),
        catalog=Catalog.load(config["data"]["catalog"]),
        agent_factory=scripted(None, {}, brains),
    )

    async def exercise() -> list[str]:
        replies = []
        for token in (None, forge_subject(alice, "mock-bob-002")):
            response = await shopping.process_exchange(user_message("hi", token))
            replies.append(response.extract_text(language=None))
        # A conversation started by Alice cannot be resumed with Bob's credential.
        first = await shopping.process_exchange(user_message("hi", alice))
        switched = user_message("hi again", bob)
        switched.echo_tokens_from(first)
        response = await shopping.process_exchange(switched)
        replies.append(response.extract_text(language=None))
        return replies

    missing, forged, switched = asyncio.run(exercise())
    assert "Authentication required" in missing
    assert "Authentication required" in forged
    assert "different user" in switched
    # Only Alice's valid first message ever reached the LLM.
    assert len(brains) == 1 and brains[0].prompts == ["hi"]
