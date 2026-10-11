"""Read-only product catalog shared by the shopping and payment agents."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SUMMARY_FIELDS = ("id", "name", "category", "price", "seller")


class Catalog:
    def __init__(self, products: list[dict[str, Any]]) -> None:
        self._products = {product["id"]: product for product in products}

    @classmethod
    def load(cls, path: str | Path) -> "Catalog":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def get(self, product_id: str) -> dict[str, Any] | None:
        product = self._products.get(product_id.strip())
        return dict(product) if product is not None else None

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Rank products by how many query words appear in their name or category."""
        words = set(re.findall(r"[a-z0-9]+", query.casefold()))
        scored = []
        for product in self._products.values():
            haystack = f"{product['name']} {product['category']}".casefold()
            score = sum(word in haystack for word in words)
            if score:
                scored.append((score, product))
        scored.sort(key=lambda item: -item[0])
        return [
            {field: product[field] for field in SUMMARY_FIELDS}
            for _, product in scored[:limit]
        ]
