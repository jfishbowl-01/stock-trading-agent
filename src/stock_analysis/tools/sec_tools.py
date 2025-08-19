# src/stock_analysis/tools/sec_tools.py
import os
import re
import time
from typing import Any, Optional, Type, Dict

import requests
from pydantic import BaseModel, Field
from crewai_tools import RagTool
from sec_api import QueryApi
from embedchain.models.data_type import DataType
import html2text

# -----------------------------
# Config & simple memoization
# -----------------------------
_SEC_API_KEY = os.getenv("SEC_API_API_KEY")

# Cache latest filing text per (form, ticker) to cut repeat downloads
# key: f"{form}:{ticker}" -> {"text": str, "expires": epoch_seconds}
_MEMO: Dict[str, Dict[str, Any]] = {}
_MEMO_TTL = int(os.getenv("SEC_FILING_CACHE_TTL_SECONDS", "21600"))  # 6h default

# polite SEC headers
_UA = {
    "User-Agent": os.getenv(
        "SEC_USER_AGENT",
        "stock-analysis-bot/1.0 (contact: youremail@example.com)"
    ),
    "Accept-Encoding": "gzip, deflate",
}

_HTML2TEXT = html2text.HTML2Text()
_HTML2TEXT.ignore_links = False


def _memo_get(key: str) -> Optional[str]:
    item = _MEMO.get(key)
    if not item:
        return None
    if item["expires"] < time.time():
        _MEMO.pop(key, None)
        return None
    return item["text"]


def _memo_set(key: str, text: str) -> None:
    _MEMO[key] = {"text": text, "expires": time.time() + _MEMO_TTL}


def _http_get(url: str, *, timeout: int = 20, retries: int = 2) -> requests.Response:
    """GET with small retries/backoff."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=_UA, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_err = e
            # simple backoff: 0s, 2s, 4s...
            time.sleep(2 ** attempt)
    raise last_err  # bubble up after retries


def _clean_text(text: str) -> str:
    # Keep letters, numbers, $, whitespace, and newlines. Drop weird control chars.
    return re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", "", text)


def _fetch_latest_filing_text(form: str, ticker: str) -> str:
    """Fetch latest 10-K/10-Q text for ticker, preferring linkToTxt."""
    if not _SEC_API_KEY:
        raise RuntimeError("SEC_API_API_KEY is not set")

    key = f"{form}:{ticker.upper()}"
    cached = _memo_get(key)
    if cached:
        return cached

    query_api = QueryApi(api_key=_SEC_API_KEY)
    query = {
        "query": {
            "query_string": {
                "query": f'ticker:{ticker} AND formType:"{form}"'
            }
        },
        "from": "0",
        "size": "1",
        "sort": [{"filedAt": {"order": "desc"}}],
    }
    data = query_api.get_filings(query)
    filings = (data or {}).get("filings", [])
    if not filings:
        raise RuntimeError(f"No {form} filings found for {ticker}")

    filing = filings[0]
    txt_url = filing.get("linkToTxt")
    html_url = filing.get("linkToFilingDetails") or filing.get("linkToHtml")

    if txt_url:
        resp = _http_get(txt_url, timeout=25, retries=2)
        text = resp.text
    elif html_url:
        resp = _http_get(html_url, timeout=25, retries=2)
        # Convert HTML → text
        try:
            text = _HTML2TEXT.handle(resp.content.decode("utf-8", errors="ignore"))
        except Exception:
            text = resp.text
    else:
        raise RuntimeError(f"{form} filing for {ticker} had no retrievable URL")

    text = _clean_text(text)
    _memo_set(key, text)
    return text


# ---------------------------------------------------------------------
# Schemas (runtime input is ONLY 'search_query' — ticker provided at init)
# ---------------------------------------------------------------------
class _SearchSchema(BaseModel):
    """Runtime input for SEC tools."""
    search_query: str = Field(
        ...,
        description="Query text to search within the filing (keywords/phrases, comma-separated allowed).",
        json_schema_extra={"required": True},
    )


# ---------------------------------------------------------------------
# Base SEC tool with circuit breaker & shared logic
# ---------------------------------------------------------------------
class _BaseSECSnippetTool(RagTool):
    """RagTool wrapper that loads filing text at init and performs quick searches at runtime.

    IMPORTANT:
    - Do NOT expose 'ticker' in args_schema; the tool instance is initialized with a ticker.
    - Agents should call with ONLY: {'search_query': '...'}
    """

    # simple per-instance circuit breaker
    _failures: int = 0
    _fail_limit: int = int(os.getenv("SEC_TOOL_FAIL_LIMIT", "3"))

    def __init__(self, ticker: Optional[str] = None, form: str = "10-K", **kwargs):
        super().__init__(**kwargs)
        self.form = form
        self.ticker = (ticker or "").upper()

        # Load & index filing text once at tool creation
        if self.ticker:
            try:
                filing_text = _fetch_latest_filing_text(self.form, self.ticker)
                # Index into RagTool’s store (mark as TEXT)
                self.add(filing_text)
                # Make the description precise for the current ticker/form
                self.description = (
                    f"Search the latest {self.form} (SEC filing) text for {self.ticker}. "
                    f"Pass only {{'search_query': '...'}} at runtime."
                )
                # Runtime schema is ONLY search_query
                self.args_schema = _SearchSchema
                self._generate_description()
            except Exception as e:
                # Keep tool alive but note failure; agent can still proceed with other tools
                self._failures += 1
                self.description = (
                    f"Latest {self.form} text for {self.ticker} could not be loaded "
                    f"(reason: {e}). Tool will return a clear error string."
                )
                self.args_schema = _SearchSchema
                self._generate_description()

    def add(self, *args: Any, **kwargs: Any) -> None:
        # Ensure the RagTool indexes as plain text
        kwargs["data_type"] = DataType.TEXT
        super().add(*args, **kwargs)

    def _keyword_windows(self, text: str, query: str, window: int = 500) -> str:
        """Fast keyword/phrase windowing to avoid heavy embedding queries."""
        terms = [t.strip().lower() for t in query.split(",") if t.strip()]
        if not terms:
            return "No valid search terms provided."
        lower = text.lower()
        snippets = []
        for term in terms:
            idx = lower.find(term)
            if idx != -1:
                start = max(0, idx - window)
                end = min(len(text), idx + window)
                snippets.append(text[start:end])
        if not snippets:
            return "No direct matches found in filing text. Consider refining the query."
        return "\n\n---\n\n".join(snippets)

    def _run(self, search_query: str, **kwargs: Any) -> Any:
        # Circuit breaker
        if self._failures >= self._fail_limit:
            return f"Temporarily disabled after {self._failures} failures. Consider alternative sources."

        try:
            # If RagTool retrieval is configured, let it handle vector search
            # via super()._run(query=...). If that fails or returns nothing,
            # fall back to quick keyword windowing using the cached filing text.
            try:
                result = super()._run(query=search_query, **kwargs)
                if isinstance(result, str) and result.strip():
                    return result
            except Exception:
                # ignore and fallback
                pass

            # Fallback: use our cached filing text (from memo or re-fetch)
            text = _memo_get(f"{self.form}:{self.ticker}")
            if not text:
                text = _fetch_latest_filing_text(self.form, self.ticker)
            return self._keyword_windows(text, search_query)

        except Exception as e:
            self._failures += 1
            return f"ERROR: {e}"


# ---------------------------------------------------------------------
# Public tools (keep class names stable to avoid changing crew.py)
# ---------------------------------------------------------------------
class SEC10KTool(_BaseSECSnippetTool):
    """Search in the specified 10-K form (latest) for the initialized ticker."""
    name: str = "Search in the specified 10-K form"
    description: str = "Search the latest 10-K (SEC filing) text for a given ticker."
    args_schema: Type[BaseModel] = _SearchSchema

    def __init__(self, ticker: Optional[str] = None, **kwargs):
        super().__init__(ticker=ticker, form="10-K", **kwargs)


class SEC10QTool(_BaseSECSnippetTool):
    """Search in the specified 10-Q form (latest) for the initialized ticker."""
    name: str = "Search in the specified 10-Q form"
    description: str = "Search the latest 10-Q (SEC filing) text for a given ticker."
    args_schema: Type[BaseModel] = _SearchSchema

    def __init__(self, ticker: Optional[str] = None, **kwargs):
        super().__init__(ticker=ticker, form="10-Q", **kwargs)