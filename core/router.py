"""Deterministic routing on top of Laya decisions. No ML, fully unit-testable."""

from typing import Any

URGENCY_ESCALATION = 0.8
MIN_ISSUE_CONFIDENCE = 0.5


def route(decisions: dict[str, Any]) -> dict[str, Any]:
    """Map normalized decisions to a queue, team and SLA.

    Rules (first match wins):
      urgency > 0.8 AND angry AND refund -> escalation_team, priority, 2h
      issue_type == billing              -> billing_team, standard, 8h
      issue_type == technical            -> engineering_team, standard, 24h
      otherwise                          -> tier1_support, normal, 48h

    `needs_review` is set when a noul answer is unsure (None) or issue_type
    confidence is low, so a human can double-check the automated decision.
    """
    urgency = decisions["urgency"]["score"]
    angry = decisions["customer_angry"]["answer"]
    refund = decisions["mentions_refund"]["answer"]
    issue = decisions["issue_type"]["answer"]

    if urgency > URGENCY_ESCALATION and angry is True and refund is True:
        queue, team, sla = "priority", "escalation_team", 2
        reason = "High urgency + refund request + customer angry"
    elif issue == "billing":
        queue, team, sla = "standard", "billing_team", 8
        reason = "Billing issue"
    elif issue == "technical":
        queue, team, sla = "standard", "engineering_team", 24
        reason = "Technical issue"
    else:
        queue, team, sla = "normal", "tier1_support", 48
        reason = f"Default routing ({issue})"

    needs_review = (
        angry is None
        or refund is None
        or decisions["issue_type"]["confidence"] < MIN_ISSUE_CONFIDENCE
    )
    suggested = decisions["assign_to"]["answer"]
    return {
        "queue": queue,
        "team": team,
        "reason": reason,
        "sla_hours": sla,
        "needs_review": needs_review,
        "model_suggested_team": suggested,
        "team_mismatch": suggested != team,
    }
