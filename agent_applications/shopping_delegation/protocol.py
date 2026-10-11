"""NLIP message conventions shared by the shopping and payment agents."""

from __future__ import annotations

from typing import Any

from nlip_sdk import AllowedFormat, NLIPMessage, NLIPSubMessage

SHOPPING_ENTITY = "shopping"
PAYMENT_ENTITY = "payment"
# Structured submessage carrying the machine-readable part of a payment request.
PAYMENT_REQUEST_LABEL = "payment_request"
PAYMENT_ACTIONS = ("place_order", "update_shipping_address", "send_gift_card")


def build_payment_request(
    request: dict[str, Any], note: str, user_token: str
) -> NLIPMessage:
    """Build the shopping agent's request to the payment agent.

    Baseline behavior, intentionally naive: the user's own credential is forwarded,
    so the payment agent acts with the user's full privileges. This forwarding is
    what the project's security extension is designed to replace.
    """
    message = NLIPMessage.text(note or f"Payment request: {request['action']}")
    message.add_submessage(
        NLIPSubMessage(
            format=AllowedFormat.structured,
            subformat="json",
            content=request,
            label=PAYMENT_REQUEST_LABEL,
        )
    )
    message.add_authentication_token(user_token)
    return message


def extract_payment_request(message: NLIPMessage) -> dict[str, Any] | None:
    request = message.extract_field(AllowedFormat.structured, label=PAYMENT_REQUEST_LABEL)
    return request if isinstance(request, dict) else None
