"""Pydantic request/response contracts."""

from pydantic import BaseModel, Field

MAX_TEXT_CHARS = 4000  # ~1000 tokens; Laya's context window is 1024 tokens
MAX_BATCH = 50


class TicketIn(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    ticket_id: str | None = None


class BatchIn(BaseModel):
    tickets: list[TicketIn] = Field(min_length=1, max_length=MAX_BATCH)


class ChoiceDecision(BaseModel):
    answer: str
    confidence: float
    probabilities: dict[str, float]


class ScoreDecision(BaseModel):
    score: float = Field(ge=0, le=1)
    confidence: float


class NoulDecision(BaseModel):
    answer: bool | None  # None = unsure (probability between the thresholds)
    probability: float
    confidence: float


class Decisions(BaseModel):
    issue_type: ChoiceDecision
    urgency: ScoreDecision
    mentions_refund: NoulDecision
    customer_angry: NoulDecision
    assign_to: ChoiceDecision


class Routing(BaseModel):
    queue: str
    team: str
    reason: str
    sla_hours: int
    needs_review: bool
    model_suggested_team: str
    team_mismatch: bool


class TriageOut(BaseModel):
    ticket_id: str
    input: str
    decisions: Decisions
    routing: Routing
    latency_ms: float  # for batch items: batch wall time / batch size


class BatchOut(BaseModel):
    count: int
    total_latency_ms: float
    results: list[TriageOut]


class HealthOut(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    warmup_latency_ms: float | None
