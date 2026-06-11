"""
API integration tests: request in -> routing -> orchestration -> response
shape out. The agent is faked (see conftest); no settings or network needed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import fake_agent, quote_call

# ---------------------------------------------------------------------------
# Static surface
# ---------------------------------------------------------------------------


def test_health_endpoint(api_client):
    resp = api_client(fake_agent()).get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_root_serves_ui(api_client):
    resp = api_client(fake_agent()).get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")


def test_static_assets_served(api_client):
    client = api_client(fake_agent())
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200


# ---------------------------------------------------------------------------
# Main path: question in, structured answer out
# ---------------------------------------------------------------------------


def test_ask_returns_expected_shape(api_client):
    agent = fake_agent("AAPL is up today.", [quote_call("AAPL")])

    resp = api_client(agent).post("/ask", json={"query": "How is AAPL doing today?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "How is AAPL doing today?"
    assert body["intent"] == "single_stock"
    assert body["tickers"] == ["AAPL"]
    assert body["answer"] == "AAPL is up today."
    assert body["tool_calls"] == [{"tool": "get_quote", "ticker": "AAPL", "ok": True}]
    assert body["conversation_id"]


def test_follow_up_carries_conversation_history(api_client):
    """Second request with the returned ID gives the agent the prior turn."""
    agent = fake_agent("AAPL is up today.", [quote_call("AAPL")])
    client = api_client(agent)

    first = client.post("/ask", json={"query": "How is AAPL doing?"}).json()
    client.post(
        "/ask",
        json={"query": "What about its P/E?", "conversation_id": first["conversation_id"]},
    )

    history = agent.answer.await_args.kwargs["history"]
    assert [(t.query, t.answer) for t in history] == [
        ("How is AAPL doing?", "AAPL is up today.")
    ]


def test_ask_comparison_reports_both_tickers(api_client):
    agent = fake_agent("Both rose.", [quote_call("TSLA"), quote_call("F")])

    resp = api_client(agent).post("/ask", json={"query": "Compare TSLA and F"})

    body = resp.json()
    assert body["intent"] == "stock_comparison"
    assert body["tickers"] == ["TSLA", "F"]


def test_failed_ticker_still_returns_an_answer(api_client):
    """A bad ticker is explained by the agent, not surfaced as an API error."""
    agent = fake_agent("I couldn't find ZZZZQ.", [quote_call("ZZZZQ", ok=False)])

    resp = api_client(agent).post("/ask", json={"query": "How is ZZZZQ doing?"})

    assert resp.status_code == 200
    assert resp.json()["intent"] == "unknown"  # nothing was successfully fetched


def test_responses_carry_unique_request_ids(api_client):
    client = api_client(fake_agent())

    first = client.get("/health").headers.get("x-request-id")
    second = client.get("/health").headers.get("x-request-id")

    assert first and second and first != second


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------


def test_rejects_empty_question(api_client):
    resp = api_client(fake_agent()).post("/ask", json={"query": ""})
    assert resp.status_code == 422


def test_rejects_whitespace_only_question(api_client):
    resp = api_client(fake_agent()).post("/ask", json={"query": "   "})
    assert resp.status_code == 422


def test_rejects_missing_body(api_client):
    resp = api_client(fake_agent()).post("/ask", json={})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_agent_outage_returns_502_without_leaking_details(api_client):
    failing = MagicMock()
    failing.answer = AsyncMock(side_effect=RuntimeError("OpenAI timeout"))

    resp = api_client(failing).post("/ask", json={"query": "How is AAPL doing?"})

    assert resp.status_code == 502
    assert "OpenAI timeout" not in resp.json()["detail"]


def test_missing_api_keys_returns_clear_503(monkeypatch):
    """No .env / env vars: the user is told what to fix, not given a bare 500."""
    import app.main as main_module
    from app.config import Settings

    def load_settings_without_env():
        return Settings(_env_file=None)  # raises ValidationError

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.setattr(main_module, "get_settings", load_settings_without_env)

    resp = TestClient(app).post("/ask", json={"query": "How is AAPL doing?"})

    assert resp.status_code == 503
    assert "OPENAI_API_KEY" in resp.json()["detail"]
