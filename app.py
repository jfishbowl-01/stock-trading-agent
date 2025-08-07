from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import os
import json
import asyncio
import re
from datetime import datetime
import uuid

# Import your crew
from src.stock_analysis.crew import StockAnalysisCrew

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stock Analysis API - watsonx Orchestrate Compatible",
    description="AI-powered stock analysis using CrewAI with Watson X Orchestrate A2A protocol support",
    version="1.0.0"
)

# Enable CORS for watsonx Orchestrate integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for job status (use Redis in production)
job_status = {}

# === EXISTING MODELS (Keep unchanged) ===
class StockAnalysisRequest(BaseModel):
    company_stock: str
    query: Optional[str] = "Analyze this company's stock performance and investment potential"

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

# === NEW WATSON X ORCHESTRATE MODELS ===
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of message sender: user, assistant, or system")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Optional name of the sender")

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="List of messages in the conversation")
    model: Optional[str] = Field("stock-analysis-agent", description="Model identifier")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.1, description="Sampling temperature")

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, Any]] = None

# === HELPER FUNCTIONS FOR WATSON X INTEGRATION ===

def extract_stock_and_query(messages: List[ChatMessage]) -> tuple[str, str]:
    """
    Extract stock symbol from Watson X Orchestrate messages and create enhanced CrewAI query
    This is the critical function that handles the prompt handoff you mentioned
    """
    
    # Get the most recent user message
    user_messages = [msg for msg in messages if msg.role == "user"]
    if not user_messages:
        # Fallback: use all message content
        user_query = " ".join([msg.content for msg in messages])
    else:
        user_query = user_messages[-1].content.strip()
    
    logger.info(f"🔍 Processing Watson X message: '{user_query}'")
    
    # Extract stock symbol using multiple sophisticated patterns
    stock_symbol = None
    
    # Pattern 1: Direct ticker symbols
    ticker_patterns = [
        r'\$([A-Z]{1,5})\b',                    # $AAPL, $TSLA
        r'\b([A-Z]{2,5})\s+(?:stock|shares|equity)\b',  # AAPL stock, TSLA shares
        r'(?:ticker|symbol)[\s:]+([A-Z]{1,5})\b',       # ticker: AAPL
        r'(?:analyze|analysis)\s+([A-Z]{2,5})\b',       # analyze AAPL
        r'\b([A-Z]{2,5})\s+(?:investment|company)\b',   # AAPL investment
    ]
    
    for pattern in ticker_patterns:
        matches = re.findall(pattern, user_query, re.IGNORECASE)
        if matches:
            # Filter out common false positives
            valid_tickers = [
                m.upper() for m in matches 
                if len(m) >= 2 and m.upper() not in [
                    'THE', 'AND', 'FOR', 'YOU', 'ARE', 'CAN', 'GET', 'ALL', 
                    'NEW', 'NOW', 'WAY', 'MAY', 'SEE', 'HIM', 'TWO', 'HOW',
                    'ITS', 'WHO', 'OIL', 'TOP', 'WIN', 'BUY', 'USE'
                ]
            ]
            if valid_tickers:
                stock_symbol = valid_tickers[0]
                logger.info(f"✅ Found ticker symbol: {stock_symbol}")
                break
    
    # Pattern 2: Company name to ticker mapping
    if not stock_symbol:
        company_mappings = {
            # Major tech companies
            'apple': 'AAPL', 'microsoft': 'MSFT', 'alphabet': 'GOOGL', 'google': 'GOOGL',
            'amazon': 'AMZN', 'meta': 'META', 'facebook': 'META', 'tesla': 'TSLA',
            'nvidia': 'NVDA', 'netflix': 'NFLX', 'adobe': 'ADBE', 'salesforce': 'CRM',
            'oracle': 'ORCL', 'intel': 'INTC', 'amd': 'AMD', 'qualcomm': 'QCOM',
            
            # Financial services
            'jpmorgan': 'JPM', 'jp morgan': 'JPM', 'bank of america': 'BAC',
            'goldman sachs': 'GS', 'morgan stanley': 'MS', 'wells fargo': 'WFC',
            'citigroup': 'C', 'american express': 'AXP',
            
            # Healthcare & pharma
            'johnson & johnson': 'JNJ', 'pfizer': 'PFE', 'abbvie': 'ABBV',
            'merck': 'MRK', 'eli lilly': 'LLY', 'bristol myers': 'BMY',
            
            # Consumer & retail
            'walmart': 'WMT', 'target': 'TGT', 'costco': 'COST', 'home depot': 'HD',
            'mcdonalds': 'MCD', 'coca cola': 'KO', 'pepsi': 'PEP', 'nike': 'NKE',
            
            # Energy & utilities
            'exxon': 'XOM', 'chevron': 'CVX', 'conocophillips': 'COP',
            
            # Popular meme/growth stocks
            'gamestop': 'GME', 'amc': 'AMC', 'palantir': 'PLTR', 'zoom': 'ZM'
        }
        
        user_query_lower = user_query.lower()
        for company, ticker in company_mappings.items():
            if company in user_query_lower:
                stock_symbol = ticker
                logger.info(f"✅ Mapped company '{company}' to ticker: {stock_symbol}")
                break
    
    # Pattern 3: Last resort - look for any 2-5 letter capital sequences
    if not stock_symbol:
        fallback_matches = re.findall(r'\b([A-Z]{2,5})\b', user_query)
        if fallback_matches:
            # Take the first reasonable match
            potential_symbols = [
                m for m in fallback_matches 
                if m not in ['THE', 'AND', 'FOR', 'YOU', 'ARE', 'CAN', 'GET']
            ]
            if potential_symbols:
                stock_symbol = potential_symbols[0]
                logger.info(f"⚠️ Using fallback ticker: {stock_symbol}")
    
    # Default fallback for demo purposes
    if not stock_symbol:
        stock_symbol = "AAPL"
        logger.warning(f"❌ No stock symbol found, defaulting to: {stock_symbol}")
    
    # === PROMPT ENHANCEMENT - This is where the magic happens ===
    # Transform simple Watson X query into comprehensive CrewAI instructions
    
    # Analyze query intent to customize the prompt
    query_lower = user_query.lower()
    
    # Determine analysis focus based on user intent
    if any(word in query_lower for word in ['buy', 'sell', 'invest', 'purchase', 'recommend']):
        analysis_focus = "INVESTMENT_DECISION"
        focus_instruction = """
**PRIMARY FOCUS: INVESTMENT DECISION**
- Provide clear BUY/SELL/HOLD recommendation with confidence level
- Include specific entry points, target prices, and stop-loss levels
- Assess risk-reward ratio and position sizing recommendations
- Compare to alternative investment opportunities
"""
    elif any(word in query_lower for word in ['risk', 'volatility', 'safe', 'dangerous', 'beta']):
        analysis_focus = "RISK_ASSESSMENT"
        focus_instruction = """
**PRIMARY FOCUS: RISK ANALYSIS**
- Detailed risk breakdown: business, financial, market, regulatory risks
- Volatility metrics, beta analysis, and downside scenarios
- Risk mitigation strategies and hedging considerations
- Stress testing against market downturns
"""
    elif any(word in query_lower for word in ['earnings', 'revenue', 'profit', 'financial', 'metrics']):
        analysis_focus = "FINANCIAL_ANALYSIS"
        focus_instruction = """
**PRIMARY FOCUS: FINANCIAL DEEP DIVE**
- Comprehensive financial statement analysis
- Key ratio analysis and trend identification
- Peer comparison and industry benchmarking
- Management guidance and forward-looking metrics
"""
    elif any(word in query_lower for word in ['news', 'recent', 'latest', 'current', 'sentiment']):
        analysis_focus = "MARKET_RESEARCH"
        focus_instruction = """
**PRIMARY FOCUS: MARKET RESEARCH & SENTIMENT**
- Latest news impact and market sentiment analysis
- Recent analyst upgrades/downgrades and price target changes
- Social media sentiment and retail investor interest
- Upcoming catalysts and market-moving events
"""
    else:
        analysis_focus = "COMPREHENSIVE"
        focus_instruction = """
**PRIMARY FOCUS: COMPREHENSIVE ANALYSIS**
- Balanced coverage of all key investment factors
- Holistic view suitable for general investment decision-making
- Equal weight to financial, market, and qualitative factors
"""
    
    # Create the enhanced prompt for CrewAI
    enhanced_query = f"""
COMPREHENSIVE STOCK ANALYSIS REQUEST FOR: {stock_symbol}

{focus_instruction}

**REQUIRED ANALYSIS COMPONENTS:**

1. **EXECUTIVE SUMMARY**
   - Investment thesis in 2-3 sentences
   - Clear recommendation: BUY/HOLD/SELL with target price
   - Key catalysts and risks summary

2. **FINANCIAL HEALTH ASSESSMENT**
   - Latest quarterly and annual performance (10-K/10-Q analysis)
   - Key metrics: P/E, EPS growth, revenue trends, profit margins
   - Debt levels, cash position, and financial stability
   - Comparison with sector peers and historical performance

3. **MARKET POSITION & COMPETITIVE ANALYSIS**
   - Industry dynamics and competitive positioning
   - Market share trends and competitive advantages
   - Management quality and strategic direction
   - Regulatory environment and compliance issues

4. **TECHNICAL & MARKET SENTIMENT**
   - Current stock performance vs. market indices
   - Recent news, press releases, and analyst opinions
   - Institutional ownership and insider trading activity
   - Social sentiment and retail investor interest

5. **FORWARD-LOOKING ANALYSIS**
   - Upcoming earnings date and expectations
   - Product launches, strategic initiatives, or major events
   - Industry trends affecting the company
   - Economic factors and market conditions impact

6. **RISK ASSESSMENT**
   - Primary business and financial risks
   - Market volatility and beta analysis
   - Regulatory and competitive threats
   - Scenario analysis (bull/base/bear cases)

7. **INVESTMENT RECOMMENDATION**
   - Clear position: BUY/HOLD/SELL with reasoning
   - Price targets and timeline expectations
   - Position sizing and risk management advice
   - Alternative investment considerations

**ORIGINAL USER QUESTION:** "{user_query}"

**RESPONSE REQUIREMENTS:**
- Use the most recent data available
- Provide specific, actionable insights
- Include relevant numbers and metrics
- Structure response clearly with headers
- Focus on answering the user's specific question while covering essential analysis areas
- Include appropriate disclaimers about investment risks

**ANALYSIS FOCUS:** {analysis_focus}
"""
    
    logger.info(f"🚀 Created enhanced query for {stock_symbol} with focus: {analysis_focus}")
    return stock_symbol, enhanced_query.strip()

async def stream_analysis_progress(job_id: str):
    """
    Stream analysis progress to Watson X Orchestrate using Server-Sent Events (SSE)
    Provides real-time updates and a heartbeat timeout if the task never starts.
    """
    logger.info(f"🚦 [stream_analysis_progress] Streaming started for Job ID: {job_id}")

    # === 1. Initial Response ===
    initial_chunk = {
        "id": f"chatcmpl-{job_id}",
        "object": "chat.completion.chunk",
        "created": int(datetime.now().timestamp()),
        "model": "stock-analysis-agent",
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": "🔍 **Initiating Comprehensive Stock Analysis**\n\nStarting multi-agent CrewAI analysis...\n\n"
            },
            "finish_reason": None
        }]
    }
    logger.info("🚀 Sending initial chunk to Orchestrate...")
    yield f"data: {json.dumps(initial_chunk)}\n\n"
    await asyncio.sleep(0.1)

    # === 2. Stream Progress ===
    start_time = datetime.now().timestamp()
    timeout = 1200  # 20 min
    last_status = None
    update_counter = 0
    heartbeat_start = datetime.now().timestamp()
    heartbeat_timeout = 15  # seconds

    while datetime.now().timestamp() - start_time < timeout:
        job = job_status.get(job_id, {})
        current_status = job.get("status", "unknown")

        # === 3. Heartbeat: Fail if task never starts ===
        if last_status is None and current_status == "queued":
            if datetime.now().timestamp() - heartbeat_start > heartbeat_timeout:
                logger.warning(f"❌ [Heartbeat Timeout] Job {job_id} stuck in 'queued'")
                error_chunk = {
                    "id": f"chatcmpl-{job_id}",
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now().timestamp()),
                    "model": "stock-analysis-agent",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": "❌ **Error:** The analysis process never started. Please try again later.\n"},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return

        # === 4. Status Updates ===
        if current_status != last_status:
            logger.info(f"🔄 Status update: {last_status} → {current_status}")

            status_messages = {
                "queued": "📋 **Analysis Queued** - Waiting for available AI agents...",
                "running": "🤖 **Analysis Running** - AI agents researching SEC filings, market data, and news...",
                "processing": "📊 **Data Processing** - Financial analysts synthesizing findings...",
            }

            if current_status in status_messages:
                chunk = {
                    "id": f"chatcmpl-{job_id}",
                    "object": "chat.completion.chunk",
                    "created": int(datetime.now().timestamp()),
                    "model": "stock-analysis-agent",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": f"{status_messages[current_status]}\n\n"},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                last_status = current_status

        # === 5. Keepalive Update ===
        update_counter += 1
        if update_counter % 24 == 0:
            elapsed = datetime.now().timestamp() - start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)

            keepalive_chunk = {
                "id": f"chatcmpl-{job_id}",
                "object": "chat.completion.chunk",
                "created": int(datetime.now().timestamp()),
                "model": "stock-analysis-agent",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": f"⏱️ **Progress Update** - Analysis in progress... ({minutes}m {seconds}s elapsed)\n"
                    },
                    "finish_reason": None
                }]
            }
            logger.info(f"📤 Sending keepalive (elapsed {minutes}m {seconds}s)")
            yield f"data: {json.dumps(keepalive_chunk)}\n\n"

        # === 6. Check for Completion ===
        if current_status in ["completed", "failed"]:
            logger.info(f"📍 Final job status detected: {current_status}")
            break

        await asyncio.sleep(2)

    # === 7. Final Result ===
    job = job_status.get(job_id, {})
    final_status = job.get("status", "unknown")

    if final_status == "completed":
        result = job.get("result", "Analysis completed but result not available.")
        final_content = f"""
## 📈 **Stock Analysis Complete**

{result}

---
**Analysis Summary:**
- **Job ID**: {job_id}
- **Analysis Time**: {int((datetime.now().timestamp() - start_time) // 60)} minutes
- **Completed**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
    else:
        error_msg = job.get("error", "Analysis timeout or unknown error occurred")
        final_content = f"""
## ❌ **Analysis Failed**

Unfortunately, the stock analysis could not be completed:

**Error**: {error_msg}

**Troubleshooting Steps:**
1. Verify the stock ticker is valid and traded on major exchanges
2. Try again in a few minutes (temporary service issues)
3. Contact support if the problem persists

**Job ID**: {job_id}
"""

    final_chunk = {
        "id": f"chatcmpl-{job_id}",
        "object": "chat.completion.chunk",
        "created": int(datetime.now().timestamp()),
        "model": "stock-analysis-agent",
        "choices": [{
            "index": 0,
            "delta": {"content": final_content},
            "finish_reason": "stop"
        }]
    }
    logger.info("✅ Sending final result chunk")
    yield f"data: {json.dumps(final_chunk)}\n\n"

    logger.info("📤 Sending [DONE] marker")
    yield "data: [DONE]\n\n"


# === EXISTING FUNCTION (Keep unchanged) ===
async def run_stock_analysis(job_id: str, company_stock: str, query: str):
    """Background task to run stock analysis"""
    try:
        logger.info(f"Starting analysis for {company_stock} (Job: {job_id})")
        job_status[job_id]["status"] = "running"
        await asyncio.sleep(2)  # short pause to simulate staged status update
        job_status[job_id]["status"] = "processing"     
        
        # Prepare inputs for the crew
        inputs = {
            'query': query,
            'company_stock': company_stock.upper(),
        }
        
        # Initialize crew with the specific stock symbol
        crew = StockAnalysisCrew(stock_symbol=company_stock.upper())
        result = crew.crew().kickoff(inputs=inputs)
        
        # Update job status with result
        job_status[job_id].update({
            "status": "completed",
            "result": str(result),
            "completed_at": datetime.now().isoformat()
        })
        
        logger.info(f"Analysis completed for {company_stock} (Job: {job_id})")
        
    except Exception as e:
        logger.error(f"Analysis failed for {company_stock} (Job: {job_id}): {str(e)}")
        job_status[job_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })

# === NEW WATSON X ORCHESTRATE ENDPOINT ===

@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest, background_tasks: BackgroundTasks):
    """
    Watson X Orchestrate A2A-compatible chat completions endpoint
    
    This is the critical endpoint that Watson X Orchestrate calls when users ask about stocks.
    It receives messages in OpenAI chat format and converts them to CrewAI analysis requests.
    """
    try:
        logger.info(f"🎯 Watson X chat completions called with {len(request.messages)} messages")
        logger.info(f"📝 Stream requested: {request.stream}")
        
        # Extract stock symbol and create enhanced CrewAI query
        stock_symbol, enhanced_query = extract_stock_and_query(request.messages)
        
        # Create unique job for tracking
        job_id = str(uuid.uuid4())
        
        # Initialize job status (reuse existing job tracking system)
        job_status[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "company_stock": stock_symbol,
            "query": enhanced_query,
            "created_at": datetime.now().isoformat(),
            "result": None,
            "error": None,
            "completed_at": None
        }
        
        # Start background analysis using existing function
        background_tasks.add_task(run_stock_analysis, job_id, stock_symbol, enhanced_query)
        
        logger.info(f"✅ Started Watson X analysis for {stock_symbol} (Job: {job_id})")
        
        # Return streaming or non-streaming response based on request
        if request.stream:
            logger.info("📡 Returning streaming response for Watson X")
            return StreamingResponse(
                stream_analysis_progress(job_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"  # Disable nginx buffering
                }
            )
        else:
            # Non-streaming response (for testing or backup)
            logger.info("📄 Returning non-streaming response")
            return ChatCompletionResponse(
                id=f"chatcmpl-{job_id}",
                created=int(datetime.now().timestamp()),
                model="stock-analysis-agent",
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"🔍 **Stock Analysis Initiated for {stock_symbol}**\n\nI'm conducting a comprehensive analysis including:\n- SEC filings research\n- Financial metrics analysis\n- Market sentiment review\n- Investment recommendation\n\nThis typically takes 5-15 minutes. The complete analysis will be delivered when ready.\n\n*Job Reference: {job_id}*"
                    },
                    "finish_reason": "stop"
                }],
                usage={
                    "prompt_tokens": sum(len(msg.content.split()) for msg in request.messages),
                    "completion_tokens": 50,
                    "total_tokens": sum(len(msg.content.split()) for msg in request.messages) + 50
                }
            )
            
    except Exception as e:
        logger.error(f"❌ Watson X chat completions failed: {str(e)}")
        
        # Return error in proper chat completion format
        error_response = ChatCompletionResponse(
            id=f"chatcmpl-error-{uuid.uuid4()}",
            created=int(datetime.now().timestamp()),
            model="stock-analysis-agent",
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant", 
                    "content": f"❌ **Analysis Error**\n\nI encountered an error starting the stock analysis:\n\n`{str(e)}`\n\nPlease try again, or contact support if this issue persists.\n\n*For technical support, please provide this error message.*"
                },
                "finish_reason": "stop"
            }],
            usage={"prompt_tokens": 0, "completion_tokens": 30, "total_tokens": 30}
        )
        
        if request.stream:
            # Return error as streaming response
            async def error_stream():
                error_chunk = {
                    "id": error_response.id,
                    "object": "chat.completion.chunk",
                    "created": error_response.created,
                    "model": error_response.model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": error_response.choices[0]["message"]["content"]},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(error_stream(), media_type="text/event-stream")
        else:
            return error_response

# === UPDATED HEALTH ENDPOINTS ===

@app.get("/")
async def root():
    """Health check endpoint with Watson X integration info"""
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
    """Detailed health check for monitoring"""
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

# === EXISTING ENDPOINTS (Keep unchanged for backward compatibility) ===

@app.post("/analyze", response_model=StockAnalysisResponse)
async def analyze_stock(
    request: StockAnalysisRequest, 
    background_tasks: BackgroundTasks
):
    """
    Start stock analysis for a given company (Legacy endpoint)
    """
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        job_status[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "company_stock": request.company_stock.upper(),
            "query": request.query,
            "created_at": datetime.now().isoformat(),
            "result": None,
            "error": None,
            "completed_at": None
        }
        
        # Start background task
        background_tasks.add_task(
            run_stock_analysis, 
            job_id, 
            request.company_stock, 
            request.query
        )
        
        logger.info(f"Analysis queued for {request.company_stock} (Job: {job_id})")
        
        return StockAnalysisResponse(
            job_id=job_id,
            status="queued",
            message=f"Stock analysis started for {request.company_stock.upper()}"
        )
        
    except Exception as e:
        logger.error(f"Failed to start analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start analysis: {str(e)}")

@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a stock analysis job"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatusResponse(**job_status[job_id])

@app.get("/jobs")
async def list_jobs():
    """List all jobs (for debugging/monitoring)"""
    return {
        "total_jobs": len(job_status),
        "jobs": list(job_status.values())
    }

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job from memory"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del job_status[job_id]
    return {"message": f"Job {job_id} deleted successfully"}

@app.post("/analyze/sync")
async def analyze_stock_sync(request: StockAnalysisRequest):
    """
    Synchronous stock analysis (use with caution - may timeout)
    """
    try:
        logger.info(f"Starting synchronous analysis for {request.company_stock}")
        
        inputs = {
            'query': request.query,
            'company_stock': request.company_stock.upper(),
        }
        
        # Initialize crew with the specific stock symbol
        crew = StockAnalysisCrew(stock_symbol=request.company_stock.upper())
        result = crew.crew().kickoff(inputs=inputs)
        
        return {
            "company_stock": request.company_stock.upper(),
            "status": "completed",
            "result": str(result),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Synchronous analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)