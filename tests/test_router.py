import pytest

from core.model import normalize, noul_answer
from core.router import route


def decisions(
    issue="general_inquiry", urgency=0.1, angry=False, refund=False, issue_conf=0.9, suggest="tier1_support"
):
    return {
        "issue_type": {"answer": issue, "confidence": issue_conf, "probabilities": {issue: issue_conf}},
        "urgency": {"score": urgency, "confidence": 0.8},
        "mentions_refund": {"answer": refund, "probability": 0.9, "confidence": 0.9},
        "customer_angry": {"answer": angry, "probability": 0.9, "confidence": 0.9},
        "assign_to": {"answer": suggest, "confidence": 0.6, "probabilities": {suggest: 0.6}},
    }


def test_escalation_requires_all_three():
    r = route(decisions("billing", urgency=0.9, angry=True, refund=True))
    assert (r["queue"], r["team"], r["sla_hours"]) == ("priority", "escalation_team", 2)


@pytest.mark.parametrize("urgency,angry,refund", [(0.8, True, True), (0.9, False, True), (0.9, True, False)])
def test_escalation_not_triggered(urgency, angry, refund):
    # urgency must be strictly > 0.8; angry and refund must both be True
    r = route(decisions("billing", urgency=urgency, angry=angry, refund=refund))
    assert r["team"] == "billing_team"


def test_billing_technical_default():
    assert route(decisions("billing"))["sla_hours"] == 8
    assert route(decisions("technical"))["team"] == "engineering_team"
    r = route(decisions("feature_request"))
    assert (r["queue"], r["team"], r["sla_hours"]) == ("normal", "tier1_support", 48)


def test_needs_review_on_unsure_noul_or_low_confidence():
    assert route(decisions(angry=None))["needs_review"] is True
    assert route(decisions(refund=None))["needs_review"] is True
    assert route(decisions(issue_conf=0.3))["needs_review"] is True
    assert route(decisions())["needs_review"] is False


def test_team_mismatch_flag():
    assert route(decisions("billing", suggest="escalation_team"))["team_mismatch"] is True
    assert route(decisions("billing", suggest="billing_team"))["team_mismatch"] is False


def test_noul_thresholds():
    assert noul_answer(0.74) is True
    assert noul_answer(0.56) is None
    assert noul_answer(0.12) is False


def test_normalize_scales_urgency_and_noul():
    raw = {
        "answers": {
            "issue_type": {"choice": "billing", "answer_confidence": 0.9, "probabilities": {"billing": 0.9, "complaint": 0.06, "technical": 0.04, "account": 0.0}},
            "assign_to": {"choice": "billing_team", "answer_confidence": 0.6, "probabilities": {"billing_team": 0.6, "tier1_support": 0.4}},
            "urgency": {"score": 1.715, "answer_confidence": 0.72},
            "mentions_refund": {"noul": 0.74},
            "customer_angry": {"noul": 0.2},
        }
    }
    d = normalize(raw)
    assert d["urgency"]["score"] == pytest.approx(0.8575)
    assert d["mentions_refund"]["answer"] is True
    assert d["customer_angry"]["answer"] is False
    assert len(d["issue_type"]["probabilities"]) == 3
