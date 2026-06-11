import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from app.agents.insights_agent import InsightsAgent
from app.agents.orchestrator import OrchestratorError, UpstreamError, run
from app.config import Settings, get_settings
from app.logging_config import request_id_var, setup_logging
from app.models.schemas import AskRequest, AskResponse
from app.services.conversation import ConversationStore
from app.services.market_data import FinnhubProvider

setup_logging()

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Stock Insights Assistant",
    description="Ask natural-language questions about stocks.",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Tag every request with a short correlation ID, echoed in X-Request-ID."""
    rid = uuid.uuid4().hex[:8]
    token = request_id_var.set(rid)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = rid
    # The UI is three tiny files; stale cached JS against fresh HTML causes
    # confusing breakage. Make browsers revalidate on every load.
    if request.url.path == "/" or request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-cache"
    return response


# ---------------------------------------------------------------------------
# Dependency factories (easily overridden in tests)
# ---------------------------------------------------------------------------

def get_settings_checked() -> Settings:
    """Turn a missing/incomplete .env into a clear error instead of a bare 500."""
    try:
        return get_settings()
    except ValidationError as exc:
        missing = ", ".join(str(e["loc"][0]).upper() for e in exc.errors())
        raise HTTPException(
            status_code=503,
            detail=f"Server is not configured: missing {missing}. "
            "Copy .env.example to .env and add your API keys.",
        ) from exc


# One store for the process lifetime; history intentionally resets on restart
_conversations = ConversationStore()


def get_conversations() -> ConversationStore:
    return _conversations


def get_market_data(settings: Settings = Depends(get_settings_checked)):
    return FinnhubProvider(settings.finnhub_api_key)


def get_insights_agent(settings: Settings = Depends(get_settings_checked)):
    return InsightsAgent(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, tags=["insights"])
async def ask(
    request: AskRequest,
    market_data=Depends(get_market_data),
    agent=Depends(get_insights_agent),
    conversations: ConversationStore = Depends(get_conversations),
):
    try:
        result = await run(
            query=request.query,
            market_data=market_data,
            agent=agent,
            conversations=conversations,
            conversation_id=request.conversation_id,
        )
    except UpstreamError as exc:
        # External service failure — not the caller's fault
        raise HTTPException(status_code=502, detail=str(exc))
    except OrchestratorError as exc:
        # Bad input (e.g. unknown ticker); already logged by the orchestrator
        raise HTTPException(status_code=422, detail=str(exc))
    return result
