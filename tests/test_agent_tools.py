import pytest

from support_lab.agent.tools import classify_error


@pytest.mark.parametrize(
    ("message", "code", "transient"),
    [
        ("Error executing tool process_refund: invalid_amount: at most 2 decimal places",
         "invalid_amount", False),
        ("Error executing tool process_refund: requires_escalation: over the limit",
         "requires_escalation", False),
        ("Error executing tool escalate_ticket: forbidden: requires scope 'tickets:write'",
         "forbidden", False),
        # An unexpected server-side exception: the SDK hides the details.
        ("Error executing tool process_refund", None, True),
        # A code-shaped prefix that isn't one of the server's policy codes.
        ("Error executing tool check_balance: database_locked: try again", "database_locked", True),
        ("tool call failed: connection reset", None, True),
    ],
)  # fmt: skip
def test_classify_error(message, code, transient):
    outcome = classify_error(message)
    assert outcome.is_error
    assert (outcome.code, outcome.transient) == (code, transient)
