from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jeff import server


class Uniform:
    """A stand-in model that gives every option the same probability."""
    backend = "mlx"
    base_model = "Qwen/Qwen3.5-0.8B"

    def decide(self, rows):
        return [([1 / len(row["question"]["criteria"])] * len(row["question"]["criteria"]), 10) for row in rows]


def request(count: int) -> dict:
    return {"model": "jeff", "state": "Book me a flight.",
            "questions": {"q": {"type": "choice", "instructions": "Which destination?",
                                "criteria": {f"City {i}": None for i in range(count)}}}}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(server.service, "model", Uniform())
    monkeypatch.setattr(server.service, "max_options", 26)
    return TestClient(server.app)


def test_questions_up_to_the_trained_limit_are_answered(client: TestClient) -> None:
    response = client.post("/v1/systemone", json=request(26))
    assert response.status_code == 200
    assert len(response.json()["answers"]["q"]["probabilities"]) == 26


def test_questions_over_the_trained_limit_are_refused_clearly(client: TestClient) -> None:
    response = client.post("/v1/systemone", json=request(27))
    assert response.status_code == 422
    assert "27 options" in response.text and "at most 26" in response.text


def test_the_limit_is_required_in_the_checkpoint_config() -> None:
    assert server.max_options({"max_options": 255}, Path("c")) == 255
    for config in ({}, {"max_options": None}, {"max_options": 1}, {"max_options": True}, {"max_options": "26"}):
        with pytest.raises(ValueError, match="max_options"):
            server.max_options(config, Path("c"))
