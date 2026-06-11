"""
FinnhubProvider tests: our transformation of Finnhub's responses into typed
models, and the failure modes per endpoint. HTTP is stubbed at the transport
layer (httpx.MockTransport) — no network.
"""

import json

import httpx
import pytest

from app.services.market_data import MAX_NEWS_ARTICLES, MAX_PEERS, FinnhubProvider

QUOTE_PAYLOAD = {"c": 189.50, "o": 187.00, "h": 191.20, "l": 186.50, "pc": 187.00}

PROFILE_PAYLOAD = {
    "name": "Apple Inc.",
    "finnhubIndustry": "Technology",
    "marketCapitalization": 2_900_000.0,
    "exchange": "NASDAQ",
    "weburl": "https://www.apple.com",
}


def make_provider(payload, status_code=200):
    """FinnhubProvider whose HTTP layer always returns *payload*."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, content=json.dumps(payload))

    return FinnhubProvider("test-key", transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------


async def test_quote_maps_fields_and_computes_change():
    quote = await make_provider(QUOTE_PAYLOAD).get_quote("aapl")

    assert quote.ticker == "AAPL"  # normalized
    assert quote.current_price == 189.50
    assert quote.prev_close == 187.00
    assert quote.change == 2.50
    assert quote.change_pct == pytest.approx(1.3369, abs=1e-4)


async def test_quote_unknown_ticker_raises_value_error():
    # Finnhub signals unknown symbols with all-zero fields
    payload = {"c": 0, "o": 0, "h": 0, "l": 0, "pc": 0}
    with pytest.raises(ValueError):
        await make_provider(payload).get_quote("ZZZZQ")


async def test_quote_http_error_propagates():
    with pytest.raises(httpx.HTTPStatusError):
        await make_provider({}, status_code=500).get_quote("AAPL")


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


async def test_profile_maps_fields():
    profile = await make_provider(PROFILE_PAYLOAD).get_company_profile("aapl")

    assert profile.ticker == "AAPL"
    assert profile.name == "Apple Inc."
    assert profile.industry == "Technology"
    assert profile.market_cap == 2_900_000.0


async def test_profile_empty_response_raises_value_error():
    # Finnhub returns {} for unknown symbols
    with pytest.raises(ValueError):
        await make_provider({}).get_company_profile("ZZZZQ")


# ---------------------------------------------------------------------------
# Peers
# ---------------------------------------------------------------------------


async def test_peers_excludes_own_ticker_and_caps_count():
    payload = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "NFLX", "NVDA", "ORCL"]

    peers = await make_provider(payload).get_peers("AAPL")

    assert "AAPL" not in peers
    assert len(peers) == MAX_PEERS


async def test_peers_empty_raises_value_error():
    with pytest.raises(ValueError):
        await make_provider([]).get_peers("ZZZZQ")


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------


async def test_news_maps_caps_and_truncates():
    payload = [
        {
            "headline": f"Headline {i}",
            "source": "Reuters",
            "datetime": 1780000000,
            "summary": "x" * 400,
        }
        for i in range(12)
    ]

    articles = await make_provider(payload).get_company_news("AAPL")

    assert len(articles) == MAX_NEWS_ARTICLES
    assert articles[0].headline == "Headline 0"
    assert len(articles[0].summary) == 300  # truncated


async def test_news_skips_items_without_headlines():
    payload = [{"summary": "no headline"}, {"headline": "Real story", "datetime": 1780000000}]

    articles = await make_provider(payload).get_company_news("AAPL")

    assert [a.headline for a in articles] == ["Real story"]


async def test_news_empty_raises_value_error():
    with pytest.raises(ValueError):
        await make_provider([]).get_company_news("AAPL")


# ---------------------------------------------------------------------------
# Financials
# ---------------------------------------------------------------------------


async def test_financials_maps_fields_and_tolerates_gaps():
    payload = {"metric": {"peTTM": 28.5, "52WeekHigh": 199.62}}

    fin = await make_provider(payload).get_basic_financials("aapl")

    assert fin.ticker == "AAPL"
    assert fin.pe_ttm == 28.5
    assert fin.week52_high == 199.62
    assert fin.dividend_yield is None  # missing metrics are None, not errors


async def test_financials_empty_raises_value_error():
    with pytest.raises(ValueError):
        await make_provider({"metric": {}}).get_basic_financials("ZZZZQ")
