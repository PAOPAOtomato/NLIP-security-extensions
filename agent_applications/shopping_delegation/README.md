# Shopping Delegation

A two-agent NLIP application for studying **cross-agent privilege laundering**. A user
logs in and talks to a **shopping agent**, which delegates every payment action to a
**payment agent** over NLIP.

```
user client ──NLIP + user credential──▶ shopping agent (A) ──NLIP──▶ payment agent (B) ──▶ ledger
                                          reads untrusted              places orders,
                                          product listings             changes address,
                                                                       sends gift cards
```

**This is the undefended baseline.** The shopping agent forwards the user's own
credential to the payment agent, and the payment agent acts on any request that carries
a valid credential. It cannot tell whether the user actually asked for the action. The
project's security extension (milestone M4) replaces that forwarding.

## Layout

| Path | Purpose |
|---|---|
| `shopping_agent.py` | Agent A. Tools: `search_products`, `get_product_details`, `request_payment` |
| `payment_agent.py` | Agent B. Tools: `place_order`, `update_shipping_address`, `send_gift_card`, `get_account_summary` |
| `protocol.py` | The A→B message: text note + `structured` submessage labeled `payment_request` + forwarded `authorization` token |
| `auth/` | Mock identity provider and ID-token verification, plus per-session user binding |
| `store.py` | Simulated payment backend. Every action is appended to `runtime/ledger.jsonl` |
| `catalog.py`, `data/` | Product catalog and test accounts (`alice`, `bob`) |
| `user_client.py` | Logs in, then chats with the shopping agent |
| `config.toml` | Ports, models, auth, data paths, and enabled security extensions |

## Setup

From the repository root:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell; on macOS/Linux: source .venv/bin/activate
pip install -e ".[agents,shopping,test]"
```

Both agents use the OpenAI API. Set `OPENAI_API_KEY` in your environment first. Models
are set per agent in `config.toml` (default `gpt-6-luna`).

## Run

Open three terminals at the repository root, with the virtual environment active.

```bash
python -m agent_applications.shopping_delegation.payment_agent    # terminal 1, port 8022
python -m agent_applications.shopping_delegation.shopping_agent   # terminal 2, port 8021
python -m agent_applications.shopping_delegation.user_client --user alice   # terminal 3
```

The user client can also run a fixed script, which is useful for demos:

```bash
python -m agent_applications.shopping_delegation.user_client --user alice \
    --message "I need a USB-C charger under 30 dollars." \
    --message "Buy one VoltMax 30W charger."
```

Every payment action appears as one JSON line in `runtime/ledger.jsonl`.

## Login

Login currently uses a **mock identity provider**. It issues RS256-signed ID tokens with
the same core claims as Google (`iss`, `aud`, `sub`, `email`, `name`, `iat`, `exp`), so
both agents verify mock and real tokens the same way. Its signing key is generated in
`.mock_idp/` on first login and is git-ignored. Google login is added in milestone M3.

Each request carries the user's ID token in an NLIP `authorization` token submessage.
Each agent checks its signature, issuer, audience, and expiry. Each agent also binds a
conversation to the first user who used it, so a replayed conversation token cannot
resume another user's session.

## Tests

```bash
python -m pytest agent_applications/shopping_delegation -q
```

The tests replace both LLMs with scripted stand-ins, so they make no API calls. They cover
login and token rejection, the payment ledger, and a full user → shopping agent → payment
agent purchase over NLIP.

## Adding a test case or rule

- **New product or poisoned listing:** add an entry to `data/catalog.json`.
- **New test user:** add an entry to `data/accounts.json`. The mock identity provider
  picks it up automatically.
- **New security extension:** list its name under `[entities.<name>.security]` in
  `config.toml`. See `security_extensions/README.md`.
