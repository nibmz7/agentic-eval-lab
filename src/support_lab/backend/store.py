"""SQLite-backed customer/account store.

This is the "environment" the agent acts on. It is deliberately the single
source of truth for policy: the MCP layer and the agent may add checks, but the
backend must refuse an invalid mutation on its own. That lets the eval suite
assert on final database state (tau-bench style) rather than trusting what the
agent *says* it did.

Money is stored as integer cents to avoid float rounding.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Refunds above this must be escalated to a human instead of processed.
REFUND_AUTO_APPROVE_LIMIT_CENTS = 500_00

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id        TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    status         TEXT NOT NULL CHECK (status IN ('active', 'suspended')),
    balance_cents  INTEGER NOT NULL CHECK (balance_cents >= 0)
);
CREATE TABLE IF NOT EXISTS refunds (
    refund_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          TEXT NOT NULL REFERENCES users(user_id),
    amount_cents     INTEGER NOT NULL CHECK (amount_cents > 0),
    idempotency_key  TEXT UNIQUE,
    created_at       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL REFERENCES users(user_id),
    reason      TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'open',
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
    entry_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    action     TEXT NOT NULL,
    payload    TEXT NOT NULL,
    outcome    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

SEED_USERS = [
    ("u_alice", "Alice Tan", "active", 120_00),
    ("u_bob", "Bob Lim", "active", 1_500_00),
    ("u_carol", "Carol Ng", "suspended", 80_00),
    ("u_dave", "Dave Koh", "active", 0),
]


class UnknownUser(LookupError):
    def __init__(self, user_id: str):
        super().__init__(f"unknown user_id: {user_id!r}")
        self.user_id = user_id


class PolicyViolation(Exception):
    """A request the backend refuses. `code` is stable and machine-readable."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Account:
    user_id: str
    name: str
    status: str
    balance_cents: int


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SupportStore:
    def __init__(self, path: str | Path = ":memory:"):
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)

    @classmethod
    def seeded(cls, path: str | Path = ":memory:") -> SupportStore:
        store = cls(path)
        with store._conn:
            store._conn.executemany(
                "INSERT OR IGNORE INTO users VALUES (?, ?, ?, ?)", SEED_USERS
            )
        return store

    def close(self) -> None:
        self._conn.close()

    # -- reads -----------------------------------------------------------

    def get_account(self, user_id: str) -> Account:
        row = self._conn.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row is None:
            raise UnknownUser(user_id)
        return Account(**dict(row))

    def snapshot(self) -> dict[str, Any]:
        """Order-stable dump of business state, for end-state assertions.

        Excludes timestamps and the audit log so two runs that reach the same
        outcome compare equal.
        """
        users = self._conn.execute(
            "SELECT user_id, status, balance_cents FROM users ORDER BY user_id"
        ).fetchall()
        refunds = self._conn.execute(
            "SELECT user_id, amount_cents FROM refunds ORDER BY refund_id"
        ).fetchall()
        tickets = self._conn.execute(
            "SELECT user_id, status FROM tickets ORDER BY ticket_id"
        ).fetchall()
        return {
            "users": [dict(r) for r in users],
            "refunds": [dict(r) for r in refunds],
            "tickets": [dict(r) for r in tickets],
        }

    def audit_log(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM audit_log ORDER BY entry_id")
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]

    # -- mutations -------------------------------------------------------

    def process_refund(
        self, user_id: str, amount_cents: int, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        payload = {
            "user_id": user_id,
            "amount_cents": amount_cents,
            "idempotency_key": idempotency_key,
        }
        try:
            result = self._process_refund(user_id, amount_cents, idempotency_key)
        except (PolicyViolation, UnknownUser) as exc:
            code = getattr(exc, "code", "unknown_user")
            self._audit("process_refund", payload, f"rejected:{code}")
            raise
        self._audit("process_refund", payload, "replayed" if result["replayed"] else "ok")
        return result

    def _process_refund(
        self, user_id: str, amount_cents: int, idempotency_key: str | None
    ) -> dict[str, Any]:
        if idempotency_key is not None:
            prior = self._conn.execute(
                "SELECT * FROM refunds WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
            if prior is not None:
                if prior["user_id"] != user_id or prior["amount_cents"] != amount_cents:
                    raise PolicyViolation(
                        "idempotency_conflict",
                        "idempotency_key was already used for a different refund",
                    )
                account = self.get_account(user_id)
                return {
                    "refund_id": prior["refund_id"],
                    "new_balance_cents": account.balance_cents,
                    "replayed": True,
                }

        account = self.get_account(user_id)
        if account.status != "active":
            raise PolicyViolation("account_not_active", f"account is {account.status}")
        if amount_cents <= 0:
            raise PolicyViolation("invalid_amount", "refund amount must be positive")
        if amount_cents > REFUND_AUTO_APPROVE_LIMIT_CENTS:
            raise PolicyViolation(
                "requires_escalation",
                "refunds over the auto-approve limit must be escalated to a human",
            )
        if amount_cents > account.balance_cents:
            raise PolicyViolation("insufficient_balance", "refund exceeds account balance")

        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO refunds (user_id, amount_cents, idempotency_key, created_at)"
                " VALUES (?, ?, ?, ?)",
                (user_id, amount_cents, idempotency_key, _now()),
            )
            self._conn.execute(
                "UPDATE users SET balance_cents = balance_cents - ? WHERE user_id = ?",
                (amount_cents, user_id),
            )
        return {
            "refund_id": cur.lastrowid,
            "new_balance_cents": account.balance_cents - amount_cents,
            "replayed": False,
        }

    def escalate_ticket(self, user_id: str, reason: str) -> dict[str, Any]:
        payload = {"user_id": user_id, "reason": reason}
        try:
            self.get_account(user_id)
            if not reason.strip():
                raise PolicyViolation("invalid_reason", "escalation reason must not be empty")
        except (PolicyViolation, UnknownUser) as exc:
            self._audit("escalate_ticket", payload, f"rejected:{getattr(exc, 'code', 'unknown_user')}")
            raise
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO tickets (user_id, reason, created_at) VALUES (?, ?, ?)",
                (user_id, reason.strip(), _now()),
            )
        self._audit("escalate_ticket", payload, "ok")
        return {"ticket_id": cur.lastrowid, "status": "open"}

    def _audit(self, action: str, payload: dict[str, Any], outcome: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO audit_log (action, payload, outcome, created_at) VALUES (?, ?, ?, ?)",
                (action, json.dumps(payload, sort_keys=True), outcome, _now()),
            )
