"""Log in as a test user and chat with the shopping agent over NLIP.

Interactive:   python -m agent_applications.shopping_delegation.user_client --user alice
Scripted:      python -m agent_applications.shopping_delegation.user_client \
                   --user alice --message "Find a USB-C charger" --message "Buy the cheapest"
"""

from __future__ import annotations

import argparse
import asyncio

import httpx

from nlip_sdk import NLIPClient, NLIPClientSession, NLIPMessage

from .auth import build_identity_provider
from .protocol import PAYMENT_ENTITY, SHOPPING_ENTITY
from .settings import load_settings


async def ask(session: NLIPClientSession, token: str, text: str) -> str:
    message = NLIPMessage.text(text)
    # Every request carries the user's credential in an NLIP authorization token.
    message.add_authentication_token(token)
    try:
        response = await session.send(message)
    except httpx.HTTPError as error:
        return f"[request failed: {error}]"
    return response.extract_text(language=None) or response.to_json()


async def run(username: str, scripted: list[str]) -> None:
    config = load_settings()
    identity_provider = build_identity_provider(config)
    token = identity_provider.login(username)
    print(f"Logged in as {username} through the mock identity provider.")

    # The user talks only to the shopping agent; payments go through it.
    client = NLIPClient.from_config(config, exclude={PAYMENT_ENTITY}, identity="user")
    session = client.create_session(SHOPPING_ENTITY)

    if scripted:
        for text in scripted:
            print(f"\nYou: {text}")
            print(f"Shopping agent: {await ask(session, token, text)}")
        return

    print("Commands: /new starts a new conversation, /exit quits.")
    while True:
        try:
            text = input("\nYou: ").strip()
        except EOFError:
            break
        if text in {"/exit", "/quit"}:
            break
        if text == "/new":
            session = client.create_session(SHOPPING_ENTITY)
            print("Started a new conversation.")
            continue
        if text:
            print(f"Shopping agent: {await ask(session, token, text)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--user", default="alice", help="Test user to log in as.")
    parser.add_argument(
        "--message",
        action="append",
        default=[],
        help="Send this message instead of chatting interactively. Repeatable.",
    )
    args = parser.parse_args()
    asyncio.run(run(args.user, args.message))


if __name__ == "__main__":
    main()
