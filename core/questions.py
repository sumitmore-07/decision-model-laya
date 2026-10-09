"""Typed question definitions for Laya (laya==0.4.1 schema).

Laya's schema: every question has `type` and `instructions`. `choice` takes a
dict of label -> description; `score` takes an ordered list of level anchors.
"""

from typing import Any

# Score answers come back on 0..(len(criteria)-1); we normalize to 0..1.
URGENCY_CRITERIA: list[str] = [
    "routine, no time pressure",
    "needs a reply soon",
    "critical, customer threatening to cancel or escalate",
]
URGENCY_MAX_LEVEL: int = len(URGENCY_CRITERIA) - 1

TRIAGE_QUESTIONS: dict[str, dict[str, Any]] = {
    "issue_type": {
        "type": "choice",
        "instructions": "What type of support issue is this?",
        "criteria": {
            "billing": "charges, invoices, payments, refunds",
            "technical": "bugs, outages, errors, something not working",
            "account": "login, password, profile, access, permissions",
            "feature_request": "asks for new functionality",
            "complaint": "dissatisfaction with service or staff",
            "general_inquiry": "questions about pricing, plans, how things work",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this support ticket?",
        "criteria": URGENCY_CRITERIA,
    },
    "mentions_refund": {
        "type": "noul",
        "instructions": "Does the customer explicitly mention wanting a refund or chargeback?",
    },
    "customer_angry": {
        "type": "noul",
        "instructions": "Is the customer expressing frustration, anger, or strong dissatisfaction?",
    },
    "assign_to": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {
            "tier1_support": "general questions and simple requests",
            "billing_team": "billing, payments, refunds",
            "engineering_team": "bugs, outages, technical failures",
            "account_management": "account access and plan changes",
            "escalation_team": "angry or high-risk customers, cancellation threats",
        },
    },
}
