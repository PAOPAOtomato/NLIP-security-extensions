"""Simulated payment backend: user accounts plus an append-only action ledger.

No real money moves. The ledger records every privileged action the payment agent
performs, which is how evaluations decide whether an attack succeeded.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from .catalog import Catalog

MAX_QUANTITY = 10
MAX_GIFT_CARD = 500.0


class StoreError(ValueError):
    """Raised when a requested payment action is invalid."""


class PaymentStore:
    def __init__(
        self, accounts: dict[str, dict[str, Any]], catalog: Catalog, ledger_path: Path
    ) -> None:
        # Accounts are keyed by the identity provider's stable subject id.
        self._accounts = {account["sub"]: dict(account) for account in accounts.values()}
        for account in self._accounts.values():
            account.setdefault("orders", [])
        self.catalog = catalog
        self.ledger_path = Path(ledger_path)
        self._lock = Lock()

    @classmethod
    def load(
        cls, accounts_path: str | Path, catalog: Catalog, ledger_path: str | Path
    ) -> "PaymentStore":
        accounts = json.loads(Path(accounts_path).read_text(encoding="utf-8"))
        return cls(accounts, catalog, Path(ledger_path))

    def _account(self, sub: str) -> dict[str, Any]:
        account = self._accounts.get(sub)
        if account is None:
            raise StoreError(f"no payment account for user {sub!r}")
        return account

    def has_account(self, sub: str) -> bool:
        return sub in self._accounts

    def summary(self, sub: str) -> dict[str, Any]:
        account = self._account(sub)
        return {
            "name": account["name"],
            "card": account["card"],
            "shipping_address": account["shipping_address"],
            "recent_orders": account["orders"][-5:],
        }

    def place_order(
        self,
        sub: str,
        product_id: str,
        quantity: int = 1,
        ship_to: str | None = None,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        account = self._account(sub)
        product = self.catalog.get(product_id)
        if product is None:
            raise StoreError(f"unknown product {product_id!r}")
        if not 1 <= quantity <= MAX_QUANTITY:
            raise StoreError(f"quantity must be between 1 and {MAX_QUANTITY}")
        order = {
            "order_id": f"ord-{uuid4().hex[:8]}",
            "product_id": product["id"],
            "product_name": product["name"],
            "quantity": quantity,
            "total": round(product["price"] * quantity, 2),
            "charged_to": account["card"],
            "ship_to": ship_to or account["shipping_address"],
        }
        account["orders"].append(order)
        self._record(sub, "place_order", order, session_id)
        return order

    def update_shipping_address(
        self, sub: str, address: str, *, session_id: str | None = None
    ) -> dict[str, Any]:
        account = self._account(sub)
        if not address.strip():
            raise StoreError("address cannot be empty")
        change = {
            "old_address": account["shipping_address"],
            "new_address": address.strip(),
        }
        account["shipping_address"] = address.strip()
        self._record(sub, "update_shipping_address", change, session_id)
        return change

    def send_gift_card(
        self,
        sub: str,
        recipient_email: str,
        amount: float,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        account = self._account(sub)
        if "@" not in recipient_email:
            raise StoreError("recipient must be an email address")
        if not 0 < amount <= MAX_GIFT_CARD:
            raise StoreError(f"gift card amount must be between 0 and {MAX_GIFT_CARD}")
        gift = {
            "gift_card_id": f"gc-{uuid4().hex[:8]}",
            "recipient_email": recipient_email.strip(),
            "amount": round(amount, 2),
            "charged_to": account["card"],
        }
        self._record(sub, "send_gift_card", gift, session_id)
        return gift

    def ledger(self) -> list[dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        lines = self.ledger_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def _record(
        self, sub: str, action: str, details: dict[str, Any], session_id: str | None
    ) -> None:
        entry = {
            "time": datetime.now(timezone.utc).isoformat(),
            "principal": sub,
            "action": action,
            "details": details,
            "nlip_session": session_id,
        }
        with self._lock:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("a", encoding="utf-8") as ledger:
                ledger.write(json.dumps(entry) + "\n")
