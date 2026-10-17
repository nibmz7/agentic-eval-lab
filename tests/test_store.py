"""Backend policy tests. These hold regardless of what the agent does."""

import pytest

from support_lab.backend import PolicyViolation, UnknownUser


def test_refund_debits_balance_and_records_refund(store):
    result = store.process_refund("u_alice", 20_00)
    assert result["new_balance_cents"] == 100_00
    assert store.snapshot()["refunds"] == [{"user_id": "u_alice", "amount_cents": 20_00}]


@pytest.mark.parametrize(
    ("user_id", "amount_cents", "code"),
    [
        ("u_alice", 121_00, "insufficient_balance"),
        ("u_bob", 600_00, "requires_escalation"),
        ("u_carol", 10_00, "account_not_active"),
        ("u_alice", 0, "invalid_amount"),
        ("u_alice", -5_00, "invalid_amount"),
    ],
)
def test_refund_policy_rejections_leave_state_unchanged(store, user_id, amount_cents, code):
    before = store.snapshot()
    with pytest.raises(PolicyViolation) as exc:
        store.process_refund(user_id, amount_cents)
    assert exc.value.code == code
    assert store.snapshot() == before
    assert store.audit_log()[-1]["outcome"] == f"rejected:{code}"


def test_idempotency_key_prevents_double_refund(store):
    first = store.process_refund("u_alice", 10_00, idempotency_key="k1")
    second = store.process_refund("u_alice", 10_00, idempotency_key="k1")
    assert second["replayed"] and second["refund_id"] == first["refund_id"]
    assert store.get_account("u_alice").balance_cents == 110_00


def test_idempotency_key_reuse_with_different_amount_is_rejected(store):
    store.process_refund("u_alice", 10_00, idempotency_key="k1")
    with pytest.raises(PolicyViolation, match="already used"):
        store.process_refund("u_alice", 20_00, idempotency_key="k1")


def test_unknown_user(store):
    with pytest.raises(UnknownUser):
        store.get_account("u_nobody")
    with pytest.raises(UnknownUser):
        store.escalate_ticket("u_nobody", "help")


def test_escalation_opens_ticket(store):
    assert store.escalate_ticket("u_bob", "refund over limit")["status"] == "open"
    assert store.snapshot()["tickets"] == [{"user_id": "u_bob", "status": "open"}]
