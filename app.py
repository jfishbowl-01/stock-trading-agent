# app.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from collections import defaultdict
import httpx
import logging
import os
import json
import asyncio
import re
from datetime import datetime
import uuid
import time

# Import your crew
from src.stock_analysis.crew import StockAnalysisCrew

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Stock Analysis API",
    description="AI-powered stock analysis using CrewAI with watsonx Orchestrate integration",
    version="1.0.0",
    openapi_version="3.0.3"  # watsonx requires OpenAPI 3.0.x
)

# CORS (tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# watsonx Configuration
# ------------------------------------------------------------------------------
FORCE_WATSON_X_SYNC = os.getenv("FORCE_WATSON_X_SYNC", "1") == "1"
FORCE_SYNC_FOR_UA_SUBSTR = os.getenv("FORCE_SYNC_FOR_UA_SUBSTR", "")
ALLOW_HEADER_FORCE_SYNC = os.getenv("ALLOW_HEADER_FORCE_SYNC", "1") == "1"

# ------------------------------------------------------------------------------
# In-memory job store & cache (use Redis in prod)
# ------------------------------------------------------------------------------
job_status: Dict[str, Dict[str, Any]] = {}

RESULT_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "21600"))  # 6h default
CALLBACK_TIMEOUT_SECONDS = float(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "10"))
ANALYSIS_TIMEOUT_SECONDS = int(os.environ.get("ANALYSIS_TIMEOUT", "900"))  # 15 minutes

# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------
class StockAnalysisRequest(BaseModel):
    company_stock: str
    query: Optional[str] = "Analyze this company's stock performance and investment potential"
    callback_url: Optional[str] = None

class StockAnalysisResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    callback_url: Optional[str] = None

# --- watsonx compatible chat payloads ---
class ChatMessage(BaseModel):
    role: str = Field(..., description="user | assistant | system")
    content: str = Field(..., description="Message content")
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = "stock-analysis-agent"
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 0.1
    callback_url: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, Any]] = None

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _cache_key(ticker: str, query: str) -> str:
    day = datetime.utcnow().strftime("%Y-%m-%d")
    return f"{ticker.upper()}:{day}:{hash(query) % 10_000_000}"

async def _post_callback(url: str, payload: Dict[str, Any], attempts: int = 3, timeout: float = None) -> None:
    """Resilient async callback with retries."""
    t = timeout or CALLBACK_TIMEOUT_SECONDS
    last_err = None
    for i in range(attempts):
        try:
            async with httpx.AsyncClient(timeout=t, follow_redirects=True) as client:
                r = await client.post(url, json=payload)
            if 200 <= r.status_code < 300:
                logger.info(f"✅ Callback delivered to {url} (HTTP {r.status_code})")
                return
            logger.warning(f"⚠️ Callback HTTP {r.status_code} to {url}: {r.text[:300]}")
        except Exception as e:
            last_err = e
            logger.error(f"❌ Callback attempt {i+1} to {url} failed: {e}")
        await asyncio.sleep(2 ** i)  # 0s, 2s, 4s
    logger.error(f"🚨 Callback failed after {attempts} attempts to {url}: {last_err}")

def extract_stock_and_query(messages: List[ChatMessage]) -> tuple[str, str]:
    """Extract ticker and create analysis query from chat messages."""
    user_messages = [m for m in messages if m.role == "user"]
    user_query = user_messages[-1].content.strip() if user_messages else ""
    logger.info(f"🔍 Processing message: '{user_query}'")

    stock_symbol = None
    
    # Ticker extraction patterns
    ticker_patterns = [
        r'\$([A-Z]{1,5})\b',           # $AAPL
        r'\b([A-Z]{2,5})\b',           # GOOG, MSFT, etc.
    ]
    
    for pattern in ticker_patterns:
        matches = re.findall(pattern, user_query, re.IGNORECASE)
        if matches:
            # Filter out common words that aren't stock tickers
            bad_words = {'THE','AND','FOR','YOU','ARE','CAN','GET','ALL','NEW','NOW','WAY','MAY','SEE','HIM','TWO','HOW','ITS','WHO','OIL','TOP','WIN','BUY','USE'}
            valid = [m.upper() for m in matches if len(m) >= 2 and m.upper() not in bad_words]
            if valid:
                stock_symbol = valid[0]
                break

    # Company name mapping
    if not stock_symbol:
        mapping = {
            'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL', 'alphabet': 'GOOGL',
            'amazon': 'AMZN', 'meta': 'META', 'facebook': 'META', 'tesla': 'TSLA',
            'nvidia': 'NVDA', 'netflix': 'NFLX', 'adobe': 'ADBE', 'jpmorgan': 'JPM',
            'starbucks': 'SBUX', 'walmart': 'WMT', 'disney': 'DIS'
        }
        query_lower = user_query.lower()
        for company_name, ticker in mapping.items():
            if company_name in query_lower:
                stock_symbol = ticker
                break

    # Default fallback
    if not stock_symbol:
        stock_symbol = "AAPL"
        logger.warning(f"❌ No stock symbol found, defaulting to: {stock_symbol}")

    # Create comprehensive analysis query
    analysis_query = f"Analyze {stock_symbol} stock and provide investment recommendation"
    
    logger.info(f"🚀 Extracted: {stock_symbol} -> {analysis_query}")
    return stock_symbol, analysis_query

# ------------------------------------------------------------------------------
# watsonx Detection
# ------------------------------------------------------------------------------
def is_watson_x_request(user_agent: str, headers: Dict[str, str]) -> bool:
    """Detect if request is from watsonx Orchestrate"""
    if not user_agent:
        return False
    
    ua_lower = user_agent.lower()
    watson_indicators = [
        'watson', 'orchestrate', 'ibm', 'watsonx', 'wxo'
    ]
    
    for indicator in watson_indicators:
        if indicator in ua_lower:
            return True
    
    # Check for IBM-specific headers
    ibm_headers = ['x-ibm', 'x-watson', 'x-orchestrate']
    for header_name in headers.keys():
        if any(ibm_header in header_name.lower() for ibm_header in ibm_headers):
            return True
    
    return False

def clean_response_content(content: str) -> str:
    """Clean response content for watsonx Orchestrate compatibility."""
    if not content:
        return "Analysis completed but no content was generated."
    
    # Remove null bytes and control characters that can break JSON
    content = content.replace('\x00', '')
    content = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', content)
    
    # Remove excessive whitespace
    content = re.sub(r'\n{3,}', '\n\n', content)
    content = re.sub(r' {2,}', ' ', content)
    
    # Ensure content is not empty after cleaning
    content = content.strip()
    if not content or len(content) < 10:
        return "Analysis completed but content was not properly formatted."
    
    return content

def format_watsonx_response(content: str, stock_symbol: str) -> str:
    """Format response content specifically for watsonx Orchestrate display."""
    
    # Clean the content first
    content = clean_response_content(content)
    
    # Add header if not present
    if not content.startswith(f"**{stock_symbol}") and not content.startswith("✅"):
        content = f"**Stock Analysis Complete for {stock_symbol}**\n\n{content}"
    
    # Ensure reasonable length for watsonx display (8000 char limit)
    if len(content) > 8000:
        content = content[:7500] + "\n\n[Analysis truncated for display. Full report available via direct API call.]"
    
    # Extract and format key sections if this is a CrewAI result
    if "Final Answer:" in content:
        # Extract the final answer section
        final_answer_match = re.search(r"Final Answer:\s*(.*?)(?:\n\n|\Z)", content, re.DOTALL)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            if final_answer:
                content = f"**Investment Analysis for {stock_symbol}**\n\n{final_answer}"
    
    return content

# ------------------------------------------------------------------------------
# Synchronous processing (for watsonx and forced sync)
# ------------------------------------------------------------------------------
async def process_immediately(payload: Dict[str, Any], source: str = "unknown") -> Dict[str, Any]:
    """Synchronous processing path for chat payloads - optimized for watsonx Orchestrate."""
    try:
        messages = payload.get("messages", [])
        if not messages:
            raise ValueError("Missing 'messages' in payload")

        logger.info(f"🚀 Starting immediate processing (source: {source})")
        stock_symbol, enhanced_query = extract_stock_and_query([ChatMessage(**m) for m in messages])
        
        # Check cache first
        key = _cache_key(stock_symbol, enhanced_query)
        cached = RESULT_CACHE.get(key)
        now_ts = datetime.now().timestamp()
        
        if cached and cached.get("expires", 0) > now_ts:
            logger.info(f"🗃️ Cache hit for {stock_symbol}")
            raw_content = cached.get("result", "")
        else:
            inputs = {'query': enhanced_query, 'company_stock': stock_symbol.upper()}
            crew = StockAnalysisCrew(stock_symbol=stock_symbol.upper())
            
            logger.info(f"🧠 Starting CrewAI analysis for {stock_symbol}")
            
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(lambda: crew.crew().kickoff(inputs=inputs)),
                    timeout=ANALYSIS_TIMEOUT_SECONDS
                )
                
                if result:
                    raw_content = str(result).strip()
                    logger.info(f"✅ CrewAI analysis completed for {stock_symbol}, length: {len(raw_content)}")
                else:
                    raw_content = f"Analysis completed for {stock_symbol} but no detailed results were generated."
                    logger.warning(f"⚠️ Empty result from CrewAI for {stock_symbol}")
                
                # Cache the result
                RESULT_CACHE[key] = {"result": raw_content, "expires": datetime.now().timestamp() + CACHE_TTL_SECONDS}
                
            except asyncio.TimeoutError:
                logger.error(f"⏳ Sync analysis timed out after {ANALYSIS_TIMEOUT_SECONDS}s for {stock_symbol}")
                raw_content = f"Analysis for {stock_symbol} timed out after {ANALYSIS_TIMEOUT_SECONDS} seconds. The analysis was too complex for real-time processing. Please try again or use a more specific query."
            except Exception as e:
                logger.error(f"❌ Sync analysis failed for {stock_symbol}: {e}")
                raw_content = f"Analysis failed for {stock_symbol}: {str(e)}. Please try again or contact support."

        # Format content for watsonx Orchestrate
        formatted_content = format_watsonx_response(raw_content, stock_symbol)
        
        logger.info(f"📊 Final response length: {len(formatted_content)} characters")
        logger.info(f"📊 Response preview: {formatted_content[:200]}...")

        # Create watsonx-compatible response
        response = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": payload.get("model", "stock-analysis-agent"),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant", 
                    "content": formatted_content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": sum(len(ChatMessage(**m).content.split()) for m in messages),
                "completion_tokens": len(formatted_content.split()),
                "total_tokens": sum(len(ChatMessage(**m).content.split()) for m in messages) + len(formatted_content.split())
            }
        }
        
        # Add metadata for debugging (not visible to watsonx)
        if source == "watson-x":
            response["_meta"] = {
                "mode": "sync", 
                "source": source,
                "stock_symbol": stock_symbol,
                "cached": cached is not None,
                "content_length": len(formatted_content)
            }
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Sync processing failed: {e}")
        
        # Return a proper error response instead of raising HTTPException
        error_content = f"I encountered an error during the stock analysis: {str(e)}. Please try again or contact support if this issue persists."
        
        error_response = {
            "id": f"chatcmpl-error-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": payload.get("model", "stock-analysis-agent"),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": error_content
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": len(error_content.split()), "total_tokens": len(error_content.split())}
        }
        return error_response

# ------------------------------------------------------------------------------
# Background worker (for non-watsonx requests)
# ------------------------------------------------------------------------------
async def run_stock_analysis(job_id: str, company_stock: str, query: str):
    logger.info(f"🏁 Background job started for {company_stock} (Job ID: {job_id})")
    try:
        key = _cache_key(company_stock, query)
        cached = RESULT_CACHE.get(key)
        now_ts = datetime.now().timestamp()
        if cached and cached.get("expires", 0) > now_ts:
            logger.info(f"🗃️ Cache hit for {company_stock}")
            job_status[job_id].update({
                "status": "completed",
                "result": cached.get("result", ""),
                "completed_at": datetime.now().isoformat()
            })
            return

        job_status[job_id]["status"] = "running"
        await asyncio.sleep(0.1)
        job_status[job_id]["status"] = "processing"

        inputs = {'query': query, 'company_stock': company_stock.upper()}
        crew = StockAnalysisCrew(stock_symbol=company_stock.upper())
        crew_instance = crew.crew()

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(lambda: crew_instance.kickoff(inputs=inputs)),
                timeout=ANALYSIS_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            logger.error(f"⏳ Analysis timed out after {ANALYSIS_TIMEOUT_SECONDS}s for {company_stock}")
            job_status[job_id].update({
                "status": "failed",
                "error": f"Analysis timed out after {ANALYSIS_TIMEOUT_SECONDS} seconds",
                "completed_at": datetime.now().isoformat()
            })
            return
        except Exception as e:
            logger.error(f"❌ Crew execution failed for {company_stock} (Job ID: {job_id}): {e}")
            job_status[job_id].update({
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now().isoformat()
            })
            return

        logger.info(f"🧠 Analysis completed for {company_stock}")

        final_str = str(result) if result else ""
        if final_str and "Final Answer" not in final_str:
            logger.warning("⚠️ CrewAI did not produce a clearly marked final answer. Using raw output.")

        RESULT_CACHE[key] = {"result": final_str, "expires": datetime.now().timestamp() + CACHE_TTL_SECONDS}

        job_status[job_id].update({
            "status": "completed",
            "result": final_str,
            "completed_at": datetime.now().isoformat()
        })
        logger.info(f"✅ Job completed for {company_stock} (Job ID: {job_id})")

    except Exception as e:
        logger.error(f"🔥 Unhandled error in run_stock_analysis for {company_stock} (Job ID: {job_id}): {e}")
        job_status[job_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })

# ------------------------------------------------------------------------------
# Async job processing (for non-watsonx requests)
# ------------------------------------------------------------------------------
async def enqueue_job(payload: Dict[str, Any], background_tasks: BackgroundTasks):
    """Async queue submission using existing background task model."""
    try:
        messages = payload.get("messages", [])
        if not messages:
            raise ValueError("Missing 'messages' in payload")

        stream = bool(payload.get("stream", False))
        callback_url = payload.get("callback_url")
        stock_symbol, enhanced_query = extract_stock_and_query([ChatMessage(**m) for m in messages])

        job_id = str(uuid.uuid4())
        job_status[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "company_stock": stock_symbol,
            "query": enhanced_query,
            "created_at": datetime.now().isoformat(),
            "result": None,
            "error": None,
            "completed_at": None,
            "callback_url": callback_url,
        }

        background_tasks.add_task(run_stock_analysis, job_id, stock_symbol, enhanced_query)
        logger.info(f"✅ Started async analysis for {stock_symbol} (Job: {job_id})")

        note = f"\n\nA callback will POST the final result to: {callback_url}" if callback_url else ""
        return ChatCompletionResponse(
            id=f"chatcmpl-{job_id}",
            created=int(datetime.now().timestamp()),
            model=payload.get("model", "stock-analysis-agent"),
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": (
                        f"🔍 **Stock Analysis Initiated for {stock_symbol}**\n\n"
                        f"Running comprehensive analysis including SEC filings, financial metrics, news sentiment, and investment recommendation.\n\n"
                        f"*Job Reference: {job_id}*{note}"
                    )
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": sum(len(ChatMessage(**m).content.split()) for m in messages),
                "completion_tokens": 50,
                "total_tokens": sum(len(ChatMessage(**m).content.split()) for m in messages) + 50
            }
        )
    except Exception as e:
        logger.error(f"❌ Enqueue failed: {e}")
        raise HTTPException(status_code=500, detail=f"Enqueue failed: {e}")

# ------------------------------------------------------------------------------
# /chat/completions (watsonx compatible) with improved routing
# ------------------------------------------------------------------------------
@app.post("/chat/completions")
async def chat_completions(
    req: Request,
    background_tasks: BackgroundTasks,
    x_force_sync: Optional[str] = Header(default=None),
    user_agent: Optional[str] = Header(default=None, alias="User-Agent"),
):
    """
    OpenAI-compatible chat completions endpoint with watsonx Orchestrate support.
    
    Automatically detects watsonx requests and routes them to synchronous processing
    to avoid background worker issues. Regular requests use async processing.
    """
    try:
        payload = await req.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    # Get request headers for debugging
    headers = dict(req.headers)
    
    # Comprehensive logging for debugging
    logger.info(f"🔍 REQUEST DEBUG:")
    logger.info(f"  User-Agent: {user_agent}")
    logger.info(f"  X-Force-Sync: {x_force_sync}")
    logger.info(f"  Headers: {list(headers.keys())}")
    logger.info(f"  Payload keys: {list(payload.keys())}")

    try:
        # Determine if we should force sync processing
        force_sync_by_header = (ALLOW_HEADER_FORCE_SYNC and x_force_sync == "1")
        force_sync_by_payload = payload.get("mode") == "sync"
        force_sync_by_ua = FORCE_SYNC_FOR_UA_SUBSTR and FORCE_SYNC_FOR_UA_SUBSTR.lower() in (user_agent or "").lower()
        force_sync_watson_x = FORCE_WATSON_X_SYNC and is_watson_x_request(user_agent or "", headers)

        force_sync = (
            force_sync_by_header or
            force_sync_by_payload or 
            force_sync_by_ua or
            force_sync_watson_x
        )

        # Log the decision
        if force_sync:
            sync_reason = []
            if force_sync_by_header: sync_reason.append("header")
            if force_sync_by_payload: sync_reason.append("payload")
            if force_sync_by_ua: sync_reason.append("user-agent")
            if force_sync_watson_x: sync_reason.append("watsonx")
            
            logger.info(f"🔄 SYNC processing engaged (reasons: {', '.join(sync_reason)})")
            result = await process_immediately(payload, source="watson-x" if force_sync_watson_x else "manual")
            return result
        else:
            logger.info("🔄 ASYNC processing engaged")
            return await enqueue_job(payload, background_tasks)

    except Exception as e:
        logger.error(f"❌ Chat completions failed: {e}")

        error_response = ChatCompletionResponse(
            id=f"chatcmpl-error-{uuid.uuid4()}",
            created=int(datetime.now().timestamp()),
            model=str((payload or {}).get("model", "stock-analysis-agent")),
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": (
                        "I encountered an error starting the stock analysis. "
                        f"Error details: {str(e)}. "
                        "Please try again or contact support if this issue persists."
                    )
                },
                "finish_reason": "stop"
            }],
            usage={"prompt_tokens": 0, "completion_tokens": 30, "total_tokens": 30}
        )
        return error_response

@app.post("/chat/completions/sync")
async def chat_completions_sync(req: Request):
    """Dedicated sync endpoint for testing"""
    try:
        payload = await req.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    logger.info("🔄 SYNC path via /sync endpoint")
    result = await process_immediately(payload, source="sync-endpoint")
    return result

# ------------------------------------------------------------------------------
# Health + Legacy endpoints
# ------------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "Stock Analysis API - watsonx Orchestrate Compatible",
        "version": "1.0.0",
        "status": "healthy",
        "watson_x_compatible": True,
        "protocols": ["OpenAPI/3.0.3"],
        "endpoints": {
            "watsonx": "/chat/completions",
            "sync": "/chat/completions/sync",
            "legacy_analyze": "/analyze",
            "legacy_status": "/status/{job_id}",
            "health": "/health"
        },
        "capabilities": [
            "stock-analysis",
            "investment-recommendations",
            "financial-analysis",
            "sec-filings-research",
            "market-sentiment-analysis",
            "watsonx-orchestrate-integration"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "stock-analysis-api",
        "version": "1.0.0",
        "watson_x_integration": {
            "enabled": True,
            "endpoint": "/chat/completions",
            "force_sync": FORCE_WATSON_X_SYNC,
            "auto_detection": True,
            "openapi_version": "3.0.3"
        },
        "active_jobs": len([j for j in job_status.values() if j.get("status") not in ["completed", "failed"]]),
        "total_jobs": len(job_status),
        "cache_entries": len(RESULT_CACHE),
        "system_info": {
            "python_version": os.sys.version.split()[0],
            "environment": os.environ.get("ENVIRONMENT", "production")
        }
    }

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/analyze", response_model=StockAnalysisResponse)
async def analyze_stock(request: StockAnalysisRequest, background_tasks: BackgroundTasks):
    try:
        job_id = str(uuid.uuid4())
        job_status[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "company_stock": request.company_stock.upper(),
            "query": request.query,
            "created_at": datetime.now().isoformat(),
            "result": None,
            "error": None,
            "completed_at": None,
            "callback_url": request.callback_url,
        }
        background_tasks.add_task(run_stock_analysis, job_id, request.company_stock, request.query)
        logger.info(f"Analysis queued for {request.company_stock} (Job: {job_id})")
        return StockAnalysisResponse(job_id=job_id, status="queued", message=f"Stock analysis started for {request.company_stock.upper()}")
    except Exception as e:
        logger.error(f"Failed to start analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start analysis: {e}")

@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(**job_status[job_id])

@app.get("/jobs")
async def list_jobs():
    return {"total_jobs": len(job_status), "jobs": list(job_status.values())}

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    del job_status[job_id]
    return {"message": f"Job {job_id} deleted successfully"}

@app.post("/analyze/sync")
async def analyze_stock_sync(request: StockAnalysisRequest):
    try:
        logger.info(f"Starting synchronous analysis for {request.company_stock}")
        inputs = {'query': request.query, 'company_stock': request.company_stock.upper()}
        crew = StockAnalysisCrew(stock_symbol=request.company_stock.upper())
        result = crew.crew().kickoff(inputs=inputs)
        return {
            "company_stock": request.company_stock.upper(),
            "status": "completed",
            "result": str(result),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Synchronous analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting Stock Analysis API on port {port}")
    logger.info(f"🔧 watsonx Sync Mode: {'ENABLED' if FORCE_WATSON_X_SYNC else 'DISABLED'}")
    logger.info(f"🔧 watsonx Auto-Detection: ENABLED")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=True)