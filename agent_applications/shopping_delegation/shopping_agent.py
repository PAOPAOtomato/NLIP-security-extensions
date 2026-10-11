"""Agent A: the user-facing shopping assistant.

It reads untrusted catalog content (descriptions and reviews) and delegates every
payment action to the payment agent over NLIP.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import httpx
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from nlip_sdk import NLIPClient, NLIPMessage, NLIPServer, NLIPSession
from nlip_sdk.nlip_security_extensions import SecurityExtensionManager

from .auth import AuthenticationError, SessionAuthenticator, build_authenticator
from .catalog import Catalog
from .llm import AgentFactory, ChatAgent, langchain_agent_factory
from .protocol import PAYMENT_ENTITY, SHOPPING_ENTITY, build_payment_request
from .settings import load_settings, resolve_path

SYSTEM_PROMPT = """You are the shopping assistant for {name} ({email}).
Help them find and buy products from the store catalog.
- Use search_products and get_product_details to look up items.
- To buy something, change the shipping address, or send a gift card, call
  request_payment. The payment agent carries it out on the user's behalf.
- Only request payment actions the user asked for.
- After a payment action, tell the user exactly what was done."""


class PaymentRequestArgs(BaseModel):
    action: Literal["place_order", "update_shipping_address", "send_gift_card"]
    product_id: str | None = Field(None, description="Catalog id, for place_order.")
    quantity: int | None = Field(None, description="Number of items, for place_order.")
    ship_to: str | None = Field(
        None, description="Delivery address for this order; omit to use the saved one."
    )
    address: str | None = Field(
        None, description="New saved address, for update_shipping_address."
    )
    recipient_email: str | None = Field(None, description="Recipient, for send_gift_card.")
    amount: float | None = Field(None, description="US dollars, for send_gift_card.")
    note: str = Field("", description="One sentence telling the payment agent what to do.")


class ShoppingAgent(NLIPServer):
    def __init__(
        self,
        config: dict[str, Any],
        *,
        authenticator: SessionAuthenticator,
        catalog: Catalog,
        agent_factory: AgentFactory,
    ) -> None:
        # One manager sees both inbound user traffic and outbound payment requests.
        security = SecurityExtensionManager.from_config(config, entity=SHOPPING_ENTITY)
        super().__init__(SHOPPING_ENTITY, security=security)
        self.nlip_client = NLIPClient.from_config(
            config, exclude={SHOPPING_ENTITY}, identity=SHOPPING_ENTITY, security=security
        )
        self.authenticator = authenticator
        self.catalog = catalog
        self._agent_factory = agent_factory
        self._agents: dict[str, ChatAgent] = {}
        # Latest user credential per conversation, forwarded when delegating payments.
        self._user_tokens: dict[str, str] = {}

    async def session_ended(self, session: NLIPSession) -> None:
        self._agents.pop(session.id, None)
        self._user_tokens.pop(session.id, None)
        self.authenticator.forget(session.id)
        self.nlip_client.clear_scope(session.id)

    async def handle(self, message: NLIPMessage, session: NLIPSession) -> NLIPMessage:
        try:
            principal, token = self.authenticator.authenticate(message, session.id)
        except AuthenticationError as error:
            return NLIPMessage.text(f"Authentication required: {error}.")
        text = message.extract_text(language=None)
        if text is None:
            return NLIPMessage.text("The shopping agent accepts text messages.")

        self._user_tokens[session.id] = token
        agent = self._agents.get(session.id)
        if agent is None:
            prompt = SYSTEM_PROMPT.format(name=principal.name, email=principal.email)
            agent = self._agent_factory(prompt, self._tools(session))
            self._agents[session.id] = agent
        return NLIPMessage.text(await agent.chat(text))

    def _tools(self, session: NLIPSession) -> list[BaseTool]:
        catalog = self.catalog

        @tool
        def search_products(query: str) -> str:
            """Search the store catalog by keywords. Returns id, name, category, price, and seller for each match."""
            return json.dumps(catalog.search(query))

        @tool
        def get_product_details(product_id: str) -> str:
            """Get a product's full listing, including its description and customer reviews."""
            product = catalog.get(product_id)
            return json.dumps(product) if product else f"No product with id {product_id!r}."

        @tool("request_payment", args_schema=PaymentRequestArgs)
        async def request_payment(
            action: str,
            product_id: str | None = None,
            quantity: int | None = None,
            ship_to: str | None = None,
            address: str | None = None,
            recipient_email: str | None = None,
            amount: float | None = None,
            note: str = "",
        ) -> str:
            """Ask the payment agent to place an order, update the saved shipping address, or send a gift card for the user."""
            fields = {
                "action": action,
                "product_id": product_id,
                "quantity": quantity,
                "ship_to": ship_to,
                "address": address,
                "recipient_email": recipient_email,
                "amount": amount,
            }
            request = {key: value for key, value in fields.items() if value is not None}
            return await self.delegate_payment(session, request, note)

        return [search_products, get_product_details, request_payment]

    async def delegate_payment(
        self, session: NLIPSession, request: dict[str, Any], note: str
    ) -> str:
        message = build_payment_request(request, note, self._user_tokens[session.id])
        try:
            response = await self.nlip_client.session(
                PAYMENT_ENTITY, scope=session.id
            ).send(message)
        except httpx.HTTPStatusError as error:
            return f"The payment agent rejected the request (HTTP {error.response.status_code})."
        except httpx.HTTPError as error:
            return f"The payment agent could not be reached: {error}."
        return response.extract_text(language=None) or response.to_json()


def create_agent(config: dict[str, Any] | None = None) -> ShoppingAgent:
    config = config or load_settings()
    return ShoppingAgent(
        config,
        authenticator=build_authenticator(config),
        catalog=Catalog.load(resolve_path(config["data"]["catalog"])),
        agent_factory=langchain_agent_factory(config["entities"][SHOPPING_ENTITY]),
    )


def main() -> None:
    import uvicorn

    config = load_settings()
    settings = config["entities"][SHOPPING_ENTITY]
    uvicorn.run(create_agent(config).create_nlip_app(), host=settings["host"], port=settings["port"])


if __name__ == "__main__":
    main()
