import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.endpoints import router
from tests.test_router import decisions


class StubModel:
    model_name = "stub"

    def __init__(self, loaded=True, fail=False):
        self.loaded = loaded
        self.fail = fail

    def predict_batch(self, texts):
        if self.fail:
            raise RuntimeError("boom")
        return [decisions("billing", urgency=0.9, angry=True, refund=True) for _ in texts]


def client(model) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.model = model
    app.state.warmup_latency_ms = 12.3
    return TestClient(app)  # no context manager: skips lifespan


def test_triage_ok():
    r = client(StubModel()).post("/triage", json={"text": "billed twice", "ticket_id": "TKT-1"})
    assert r.status_code == 200
    body = r.json()
    assert body["ticket_id"] == "TKT-1"
    assert body["routing"]["queue"] == "priority"
    assert body["latency_ms"] >= 0


def test_ticket_id_generated():
    r = client(StubModel()).post("/triage", json={"text": "hi"})
    assert r.json()["ticket_id"].startswith("TKT-")


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "x" * 4001}, {}])
def test_triage_validation(payload):
    assert client(StubModel()).post("/triage", json=payload).status_code == 422


def test_batch_ok_and_limits():
    c = client(StubModel())
    r = c.post("/batch", json={"tickets": [{"text": "a"}, {"text": "b"}]})
    assert r.status_code == 200 and r.json()["count"] == 2
    assert c.post("/batch", json={"tickets": []}).status_code == 422
    assert c.post("/batch", json={"tickets": [{"text": "a"}] * 51}).status_code == 422


def test_model_not_loaded_is_503():
    assert client(StubModel(loaded=False)).post("/triage", json={"text": "a"}).status_code == 503


def test_inference_failure_is_503_without_leak():
    r = client(StubModel(fail=True)).post("/triage", json={"text": "a"})
    assert r.status_code == 503 and "boom" not in r.text


def test_health():
    r = client(StubModel()).get("/health")
    assert r.json() == {"status": "ok", "model_loaded": True, "model_name": "stub", "warmup_latency_ms": 12.3}
    assert client(StubModel(loaded=False)).get("/health").json()["status"] == "loading"
