from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class QueryIntent(str, Enum):
    single_stock = "single_stock"
    stock_comparison = "stock_comparison"
    company_lookup = "company_lookup"
    unknown = "unknown"


class QuoteData(BaseModel):
    ticker: str
    current_price: float
    open_price: float
    high_price: float
    low_price: float
    prev_close: float
    change: float
    change_pct: float


class CompanyProfile(BaseModel):
    ticker: str
    name: str
    industry: str
    market_cap: float
    exchange: str
    logo: Optional[str] = None
    weburl: Optional[str] = None


class NewsArticle(BaseModel):
    headline: str
    source: str
    date: str
    summary: str = ""


class BasicFinancials(BaseModel):
    """Key valuation metrics; fields are optional because Finnhub coverage
    varies by company."""

    ticker: str
    pe_ttm: Optional[float] = None
    eps_ttm: Optional[float] = None
    week52_high: Optional[float] = None
    week52_low: Optional[float] = None
    dividend_yield: Optional[float] = None
    net_margin: Optional[float] = None
    beta: Optional[float] = None


class AskRequest(BaseModel):
    # Strip whitespace before min_length runs, so "   " is rejected as empty
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(..., min_length=1, max_length=500, description="Natural language question about stocks")
    conversation_id: Optional[str] = Field(
        default=None,
        max_length=32,
        description="Continue an existing conversation; omit to start a new one",
    )


class ToolCallRecord(BaseModel):
    """One tool invocation the agent made while answering."""

    tool: str
    ticker: str
    ok: bool


class AskResponse(BaseModel):
    query: str
    intent: QueryIntent
    tickers: list[str]
    answer: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    conversation_id: str
