# app.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
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
import time  # added for callback backoff

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
    title="Stock Analysis API - watsonx Orchestrate Compatible",
    description="AI-powered stock analysis using CrewAI with Watson X Orchestrate A2A protocol support",
    version="1.0.0"
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
# In-memory job store & cache (use Redis in prod)
# ------------------------------------------------------------------------------
job_status: Dict[str, Dict[str, Any]] = {}

RESULT_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "21600"))  # 6h default
CALLBACK_TIMEOUT_SECONDS = float(os.environ.get("CALLBACK_TIMEOUT_SECONDS", "10"))
ANALYSIS_TIMEOUT_SECONDS = int(os.environ.get("ANALYSIS_TIMEOUT", "600"))

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

# --- Watsonx-style chat payloads ---
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
    """Resilient async callback with small retries + logging."""
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
    """Extract ticker + build an enhanced analysis prompt for CrewAI."""
    user_messages = [m for m in messages if m.role == "user"]
    user_query = user_messages[-1].content.strip() if user_messages else " ".join(m.content for m in messages)
    logger.info(f"🔍 Processing Watson X message: '{user_query}'")

    stock_symbol = None
    ticker_patterns = [
        r'\$([A-Z]{1,5})\b',
        r'\b([A-Z]{2,5})\s+(?:stock|shares|equity)\b',
        r'(?:ticker|symbol)[\s:]+([A-Z]{1,5})\b',
        r'(?:analyze|analysis)\s+([A-Z]{2,5})\b',
        r'\b([A-Z]{2,5})\s+(?:investment|company)\b',
    ]
    for pattern in ticker_patterns:
        matches = re.findall(pattern, user_query, re.IGNORECASE)
        if matches:
            bad = {'THE','AND','FOR','YOU','ARE','CAN','GET','ALL','NEW','NOW','WAY','MAY','SEE','HIM','TWO','HOW','ITS','WHO','OIL','TOP','WIN','BUY','USE'}
            valid = [m.upper() for m in matches if len(m) >= 2 and m.upper() not in bad]
            if valid:
                stock_symbol = valid[0]; break

    if not stock_symbol:
        mapping = {
            'apple':'AAPL','microsoft':'MSFT','alphabet':'GOOGL','google':'GOOGL','amazon':'AMZN','meta':'META',
            'facebook':'META','tesla':'TSLA','nvidia':'NVDA','netflix':'NFLX','adobe':'ADBE','salesforce':'CRM',
            'oracle':'ORCL','intel':'INTC','amd':'AMD','qualcomm':'QCOM','jpmorgan':'JPM','jp morgan':'JPM',
            'bank of america':'BAC','goldman sachs':'GS','morgan stanley':'MS','wells fargo':'WFC','citigroup':'C',
            'american express':'AXP','johnson & johnson':'JNJ','pfizer':'PFE','abbvie':'ABBV','merck':'MRK',
            'eli lilly':'LLY','bristol myers':'BMY','walmart':'WMT','target':'TGT','costco':'COST','home depot':'HD',
            'mcdonalds':'MCD','coca cola':'KO','pepsi':'PEP','nike':'NKE','exxon':'XOM','chevron':'CVX','conocophillips':'COP',
            'gamestop':'GME','amc':'AMC','palantir':'PLTR','zoom':'ZM','ibm':'IBM'
        }
        ql = user_query.lower()
        for k,v in mapping.items():
            if k in ql:
                stock_symbol = v; break

    if not stock_symbol:
        fallback = [m for m in re.findall(r'\b([A-Z]{2,5})\b', user_query) if m not in {'THE','AND','FOR','YOU','ARE','CAN','GET'}]
        if fallback:
            stock_symbol = fallback[0]

    if not stock_symbol:
        stock_symbol = "AAPL"
        logger.warning(f"❌ No stock symbol found, defaulting to: {stock_symbol}")

    query_lower = user_query.lower()
    if any(w in query_lower for w in ['buy','sell','invest','purchase','recommend']):
        analysis_focus = "INVESTMENT_DECISION"
        focus_instruction = """
**PRIMARY FOCUS: INVESTMENT DECISION**
- Provide BUY/SELL/HOLD with confidence
- Entry/targets/stop-loss; risk-reward and sizing
- Compare to alternatives
"""
    elif any(w in query_lower for w in ['risk','volatility','safe','dangerous','beta']):
        analysis_focus = "RISK_ASSESSMENT"
        focus_instruction = """
**PRIMARY FOCUS: RISK ANALYSIS**
- Business/financial/market/regulatory risks
- Volatility/beta; downside scenarios
- Mitigation/hedging
"""
    elif any(w in query_lower for w in ['earnings','revenue','profit','financial','metrics']):
        analysis_focus = "FINANCIAL_ANALYSIS"
        focus_instruction = """
**PRIMARY FOCUS: FINANCIAL DEEP DIVE**
- FS/ratios/trends; peer benchmarking
- Guidance/forward-looking metrics
"""
    elif any(w in query_lower for w in ['news','recent','latest','current','sentiment']):
        analysis_focus = "MARKET_RESEARCH"
        focus_instruction = """
**PRIMARY FOCUS: MARKET RESEARCH & SENTIMENT**
- Latest news/sentiment; analysts changes
- Upcoming catalysts
"""
    else:
        analysis_focus = "COMPREHENSIVE"
        focus_instruction = """
**PRIMARY FOCUS: COMPREHENSIVE ANALYSIS**
- Balanced view across fundamentals, market, qualitative factors
"""

    enhanced_query = f"""
COMPREHENSIVE STOCK ANALYSIS REQUEST FOR: {stock_symbol}

{focus_instruction}

**REQUIRED ANALYSIS COMPONENTS:**
1) Executive summary + clear recommendation
2) Financials (P/E, EPS growth, revenue, margins, D/E) with filings
3) Competitive/market position
4) Sentiment/news/catalysts
5) Forward-looking (earnings, initiatives)
6) Risks (business/financial/market/regulatory)
7) Final recommendation with rationale

**ORIGINAL USER QUESTION:** "{user_query}"
**ANALYSIS FOCUS:** {analysis_focus}
- Use most recent verifiable data.
- Prefer concise, structured output.
"""
    logger.info(f"🚀 Created enhanced query for {stock_symbol} with focus: {analysis_focus}")
    return stock_symbol, enhanced_query.strip()

# ------------------------------------------------------------------------------
# Streaming (SSE)
# ------------------------------------------------------------------------------
async def stream_analysis_progress(job_id: str):
    logger.info(f"🚦 [stream_analysis_progress] Started for Job ID: {job_id}")

    initial_chunk = {
        "id": f"chatcmpl-{job_id}",
        "object": "chat.completion.chunk",
        "created": int(datetime.now().timestamp()),
        "model": "stock-analysis-agent",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "📡 Analysis starting...\n\n"}, "finish_reason": None}]
    }
    yield f"data: {json.dumps(initial_chunk)}\n\n"
    await asyncio.sleep(0.1)

    start_time = datetime.now().timestamp()
    timeout = 1200
    last_status = None
    update_counter = 0
    heartbeat_start = datetime.now().timestamp()
    heartbeat_timeout = 15

    while datetime.now().timestamp() - start_time < timeout:
        job = job_status.get(job_id, {})
        current_status = job.get("status", "unknown")

        if last_status is None and current_status == "queued":
            if datetime.now().timestamp() - heartbeat_start > heartbeat_timeout:
                logger.warning(f"❌ Heartbeat timeout for Job {job_id}")
                error_chunk = {
                    "id": f"chatcmpl-{job_id}",
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now().timestamp()),
                    "model": "stock-analysis-agent",
                    "choices": [{"index": 0, "delta": {"content": "❌ Error: analysis did not start. Try again later.\n"}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return

        if current_status != last_status:
            logger.info(f"🔄 Status: {last_status} → {current_status}")
            status_msg = {
                "queued": "📋 Queued…",
                "running": "🤖 Running analysis…",
                "processing": "📊 Synthesizing findings…",
            }.get(current_status)
            if status_msg:
                chunk = {
                    "id": f"chatcmpl-{job_id}",
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now().timestamp()),
                    "model": "stock-analysis-agent",
                    "choices": [{"index": 0, "delta": {"content": f"{status_msg}\n\n"}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                last_status = current_status

        update_counter += 1
        if update_counter % 24 == 0:
            elapsed = int(datetime.now().timestamp() - start_time)
            minutes, seconds = elapsed // 60, elapsed % 60
            keepalive_chunk = {
                "id": f"chatcmpl-{job_id}",
                "object": "chat.completion.chunk",
                "created": int(datetime.now().timestamp()),
                "model": "stock-analysis-agent",
                "choices": [{"index": 0, "delta": {"content": f"⏱️ Progress… ({minutes}m {seconds}s)\n"}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(keepalive_chunk)}\n\n"

        if current_status in ["completed", "failed"]:
            logger.info(f"📍 Job final status detected: {current_status}")
            break

        await asyncio.sleep(2)

    job = job_status.get(job_id, {})
    final_status = job.get("status", "unknown")

    if final_status == "completed":
        result = job.get("result", "Analysis completed but result not available.")
        final_content = f"## 📈 Analysis Complete\n\n{result}\n"
    else:
        error_msg = job.get("error", "Analysis timeout or unknown error occurred")
        final_content = f"## ❌ Analysis Failed\n\n**Error**: {error_msg}\n"

    final_payload = {
        "job_id": job_id,
        "status": final_status,
        "result": job.get("result"),
        "error": job.get("error"),
        "company_stock": job.get("company_stock"),
        "created_at": job.get("created_at"),
        "completed_at": job.get("completed_at"),
    }
    cb_url = job.get("callback_url")
    if cb_url:
        logger.info(f"📬 Posting final result to callback: {cb_url}")
        asyncio.create_task(_post_callback(cb_url, final_payload))

    final_chunk = {
        "id": f"chatcmpl-{job_id}",
        "object": "chat.completion.chunk",
        "created": int(datetime.now().timestamp()),
        "model": "stock-analysis-agent",
        "choices": [{"index": 0, "delta": {"content": final_content}, "finish_reason": "stop"}]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

# ------------------------------------------------------------------------------
# Background worker
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
            cb_url = job_status[job_id].get("callback_url")
            if cb_url:
                asyncio.create_task(_post_callback(cb_url, {
                    "job_id": job_id,
                    "status": "completed",
                    "result": cached.get("result", ""),
                    "company_stock": company_stock,
                    "created_at": job_status[job_id]["created_at"],
                    "completed_at": job_status[job_id]["completed_at"]
                }))
            return

        job_status[job_id]["status"] = "running"
        await asyncio.sleep(0)
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
            cb_url = job_status[job_id].get("callback_url")
            if cb_url:
                asyncio.create_task(_post_callback(cb_url, {
                    "job_id": job_id,
                    "status": "failed",
                    "error": f"Analysis timed out after {ANALYSIS_TIMEOUT_SECONDS} seconds",
                    "company_stock": company_stock,
                    "created_at": job_status[job_id]["created_at"],
                    "completed_at": job_status[job_id]["completed_at"]
                }))
            return
        except Exception as e:
            logger.error(f"❌ Crew execution failed for {company_stock} (Job ID: {job_id}): {e}")
            job_status[job_id].update({
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now().isoformat()
            })
            cb_url = job_status[job_id].get("callback_url")
            if cb_url:
                asyncio.create_task(_post_callback(cb_url, {
                    "job_id": job_id,
                    "status": "failed",
                    "error": str(e),
                    "company_stock": company_stock,
                    "created_at": job_status[job_id]["created_at"],
                    "completed_at": job_status[job_id]["completed_at"]
                }))
            return

        logger.info(f"🧠 Raw CrewAI result for {company_stock}: {result}")

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
        cb_url = job_status[job_id].get("callback_url")
        if cb_url:
            asyncio.create_task(_post_callback(cb_url, {
                "job_id": job_id,
                "status": "completed",
                "result": final_str,
                "company_stock": company_stock,
                "created_at": job_status[job_id]["created_at"],
                "completed_at": job_status[job_id]["completed_at"]
            }))

    except Exception as e:
        logger.error(f"🔥 Unhandled error in run_stock_analysis for {company_stock} (Job ID: {job_id}): {e}")
        job_status[job_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })
        cb_url = job_status[job_id].get("callback_url")
        if cb_url:
            asyncio.create_task(_post_callback(cb_url, {
                "job_id": job_id,
                "status": "failed",
                "error": str(e),
                "company_stock": company_stock,
                "created_at": job_status[job_id]["created_at"],
                "completed_at": job_status[job_id]["completed_at"]
            }))

# ------------------------------------------------------------------------------
# /chat/completions (Watsonx-compatible)
# ------------------------------------------------------------------------------
@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest, background_tasks: BackgroundTasks):
    try:
        logger.info(f"🎯 Watson X chat called with {len(request.messages)} messages (stream={request.stream})")
        stock_symbol, enhanced_query = extract_stock_and_query(request.messages)

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
            "callback_url": request.callback_url,
        }

        background_tasks.add_task(run_stock_analysis, job_id, stock_symbol, enhanced_query)
        logger.info(f"✅ Started Watson X analysis for {stock_symbol} (Job: {job_id})")

        if request.stream:
            return StreamingResponse(
                stream_analysis_progress(job_id),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
            )
        else:
            note = f"\n\nA callback will POST the final result to: {request.callback_url}" if request.callback_url else ""
            return ChatCompletionResponse(
                id=f"chatcmpl-{job_id}",
                created=int(datetime.now().timestamp()),
                model="stock-analysis-agent",
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": (
                            f"🔍 **Stock Analysis Initiated for {stock_symbol}**\n\n"
                            f"Running filings, financial metrics, news/sentiment, and recommendation.\n\n"
                            f"*Job Reference: {job_id}*{note}"
                        )
                    },
                    "finish_reason": "stop"
                }],
                usage={
                    "prompt_tokens": sum(len(m.content.split()) for m in request.messages),
                    "completion_tokens": 50,
                    "total_tokens": sum(len(m.content.split()) for m in request.messages) + 50
                }
            )

    except Exception as e:
        logger.error(f"❌ Watson X chat completions failed: {e}")

        error_response = ChatCompletionResponse(
            id=f"chatcmpl-error-{uuid.uuid4()}",
            created=int(datetime.now().timestamp()),
            model="stock-analysis-agent",
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": (
                        "❌ **Analysis Error**\n\n"
                        f"I encountered an error starting the stock analysis:\n\n`{str(e)}`\n\n"
                        "Please try again, or contact support if this issue persists."
                    )
                },
                "finish_reason": "stop"
            }],
            usage={"prompt_tokens": 0, "completion_tokens": 30, "total_tokens": 30}
        )

        if request.stream:
            async def error_stream():
                chunk = {
                    "id": error_response.id,
                    "object": "chat.completion.chunk",
                    "created": error_response.created,
                    "model": error_response.model,
                    "choices": [{"index": 0, "delta": {"content": error_response.choices[0]["message"]["content"]}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(error_stream(), media_type="text/event-stream")
        else:
            return error_response

# ------------------------------------------------------------------------------
# Health + Legacy
# ------------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "Stock Analysis API - Watson X Orchestrate Compatible",
        "version": "1.0.0",
        "status": "healthy",
        "watson_x_compatible": True,
        "protocols": ["A2A/0.2.1"],
        "endpoints": {
            "watson_x": "/chat/completions",
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
            "streaming-responses"
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
            "streaming_supported": True,
            "a2a_protocol": "0.2.1"
        },
        "active_jobs": len(job_status),
        "system_info": {
            "python_version": os.sys.version.split()[0],
            "environment": os.environ.get("ENVIRONMENT", "production")
        }
    }

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
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=True)