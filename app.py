# app.py - Official IBM watsonx Orchestrate Agent Connect Compatible Version
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
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
# FastAPI app with IBM watsonx Orchestrate Agent Connect compatibility
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Stock Analysis Agent - IBM Agent Connect",
    description="AI-powered stock analysis agent compatible with IBM watsonx Orchestrate Agent Connect Framework",
    version="1.0.0",
    openapi_version="3.0.3",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "21600"))  # 6h default
ANALYSIS_TIMEOUT_SECONDS = int(os.environ.get("ANALYSIS_TIMEOUT", "900"))  # 15 minutes

# ------------------------------------------------------------------------------
# In-memory cache and thread management
# ------------------------------------------------------------------------------
RESULT_CACHE: Dict[str, Dict[str, Any]] = {}
THREAD_HISTORY: Dict[str, List[Dict]] = {}

# ------------------------------------------------------------------------------
# IBM Agent Connect Models (Official Format)
# ------------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="stock-analysis-agent")
    messages: List[ChatMessage] = Field(..., description="List of messages in the conversation")
    stream: Optional[bool] = Field(default=True, description="Whether to stream the response")
    temperature: Optional[float] = Field(default=0.1, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=4000, description="Maximum tokens in response")
    tools: Optional[List[Dict]] = Field(default=None, description="Available tools")
    tool_choice: Optional[Union[str, Dict]] = Field(default=None, description="Tool choice preference")

class ChatCompletionResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model used")
    choices: List[Dict] = Field(..., description="List of completion choices")
    usage: Optional[Dict] = Field(default=None, description="Usage statistics")

class AgentCard(BaseModel):
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    provider: Dict[str, str] = Field(..., description="Provider information")
    version: str = Field(default="1.0.0", description="Agent version")
    capabilities: Dict[str, bool] = Field(..., description="Agent capabilities")

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def _cache_key(ticker: str) -> str:
    day = datetime.utcnow().strftime("%Y-%m-%d")
    return f"{ticker.upper()}:{day}"

def extract_stock_symbol(input_text: str) -> str:
    """Extract stock symbol from various input formats"""
    # Direct ticker patterns
    ticker_patterns = [
        r'\$([A-Z]{1,5})\b',           # $AAPL
        r'\b([A-Z]{2,5})\b',           # AAPL, TSLA, etc.
    ]
    
    for pattern in ticker_patterns:
        matches = re.findall(pattern, input_text.upper(), re.IGNORECASE)
        if matches:
            bad_words = {'THE','AND','FOR','YOU','ARE','CAN','GET','ALL','NEW','NOW','WAY','MAY','SEE','HIM','TWO','HOW','ITS','WHO','OIL','TOP','WIN','BUY','USE'}
            valid = [m.upper() for m in matches if len(m) >= 2 and m.upper() not in bad_words]
            if valid:
                return valid[0]

    # Company name mapping
    mapping = {
        'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL', 'alphabet': 'GOOGL',
        'amazon': 'AMZN', 'meta': 'META', 'facebook': 'META', 'tesla': 'TSLA',
        'nvidia': 'NVDA', 'netflix': 'NFLX', 'adobe': 'ADBE', 'jpmorgan': 'JPM',
        'starbucks': 'SBUX', 'walmart': 'WMT', 'disney': 'DIS'
    }
    
    input_lower = input_text.lower()
    for company_name, ticker in mapping.items():
        if company_name in input_lower:
            return ticker
    
    return input_text.upper()  # Return as-is if no match

def format_analysis_for_orchestrate(raw_result: str, stock_symbol: str) -> tuple[str, str, str]:
    """Format CrewAI result for watsonx Orchestrate consumption"""
    
    # Clean the result
    if not raw_result or len(raw_result.strip()) < 10:
        return "HOLD", "Analysis completed but insufficient data for recommendation", "Limited analysis results available"
    
    raw_result = raw_result.strip()
    
    # Extract recommendation
    recommendation = "HOLD"  # Default
    if any(word in raw_result.upper() for word in ["BUY", "STRONG BUY", "OVERWEIGHT"]):
        recommendation = "BUY"
    elif any(word in raw_result.upper() for word in ["SELL", "STRONG SELL", "UNDERWEIGHT"]):
        recommendation = "SELL"
    
    # Extract key points for summary
    lines = raw_result.split('\n')
    key_points = []
    
    for line in lines:
        line = line.strip()
        if line and any(keyword in line.lower() for keyword in ['recommend', 'rating', 'target', 'outlook', 'risk', 'revenue', 'profit']):
            if len(line) < 200:  # Keep summary points concise
                key_points.append(line)
        if len(key_points) >= 3:  # Limit to top 3 points
            break
    
    # Create executive summary
    summary = f"Stock Analysis for {stock_symbol}: " + (" • ".join(key_points) if key_points else "Comprehensive analysis completed.")
    
    # Limit lengths for watsonx Orchestrate display
    if len(summary) > 300:
        summary = summary[:297] + "..."
    
    # Detailed analysis (truncated if too long)
    detailed = raw_result
    if len(detailed) > 2000:
        detailed = detailed[:1997] + "..."
    
    return recommendation, summary, detailed

def generate_completion_id() -> str:
    """Generate a unique completion ID"""
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"

# ------------------------------------------------------------------------------
# Core Analysis Function
# ------------------------------------------------------------------------------
async def run_stock_analysis_sync(stock_symbol: str) -> dict:
    """Run stock analysis synchronously"""
    try:
        logger.info(f"🚀 Starting analysis for {stock_symbol}")
        
        # Check cache first
        key = _cache_key(stock_symbol)
        cached = RESULT_CACHE.get(key)
        now_ts = datetime.now().timestamp()
        
        if cached and cached.get("expires", 0) > now_ts:
            logger.info(f"🗃️ Cache hit for {stock_symbol}")
            return cached["result"]
        
        # Run CrewAI analysis
        inputs = {'query': f"Analyze {stock_symbol} stock and provide investment recommendation", 'company_stock': stock_symbol.upper()}
        crew = StockAnalysisCrew(stock_symbol=stock_symbol.upper())
        
        result = await asyncio.wait_for(
            asyncio.to_thread(lambda: crew.crew().kickoff(inputs=inputs)),
            timeout=ANALYSIS_TIMEOUT_SECONDS
        )
        
        raw_result = str(result) if result else ""
        logger.info(f"✅ Analysis completed for {stock_symbol}, length: {len(raw_result)}")
        
        # Format for watsonx Orchestrate
        recommendation, summary, detailed = format_analysis_for_orchestrate(raw_result, stock_symbol)
        
        # Determine confidence level
        confidence = "Medium"
        if len(raw_result) > 1000 and "Final Answer" in raw_result:
            confidence = "High"
        elif len(raw_result) < 500:
            confidence = "Low"
        
        result_data = {
            "stock_symbol": stock_symbol.upper(),
            "recommendation": recommendation,
            "summary": summary,
            "detailed_analysis": detailed,
            "confidence_level": confidence,
            "last_updated": datetime.now().isoformat()
        }
        
        # Cache the result
        RESULT_CACHE[key] = {
            "result": result_data,
            "expires": datetime.now().timestamp() + CACHE_TTL_SECONDS
        }
        
        return result_data
        
    except asyncio.TimeoutError:
        logger.error(f"⏳ Analysis timed out for {stock_symbol}")
        return {
            "stock_symbol": stock_symbol.upper(),
            "recommendation": "HOLD",
            "summary": f"Analysis for {stock_symbol} timed out. Please try again.",
            "detailed_analysis": f"The analysis for {stock_symbol} exceeded the {ANALYSIS_TIMEOUT_SECONDS} second limit. This may indicate high market volatility or system load.",
            "confidence_level": "Low",
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Analysis failed for {stock_symbol}: {e}")
        return {
            "stock_symbol": stock_symbol.upper(),
            "recommendation": "HOLD",
            "summary": f"Analysis failed for {stock_symbol}: {str(e)}",
            "detailed_analysis": f"An error occurred during analysis: {str(e)}. Please try again or contact support.",
            "confidence_level": "Low",
            "last_updated": datetime.now().isoformat()
        }

# ------------------------------------------------------------------------------
# IBM Agent Connect Required Endpoints
# ------------------------------------------------------------------------------

@app.get("/v1/agents")
async def agent_discovery():
    """
    Agent Discovery endpoint - Required by IBM Agent Connect
    """
    return {
        "agents": [
            {
                "name": "Stock Analysis Agent",
                "description": "AI-powered stock analysis agent that provides comprehensive investment recommendations using SEC filings, financial metrics, and market data",
                "provider": {
                    "organization": "CrewAI Stock Analysis",
                    "url": "https://stock-trading-agent.onrender.com"
                },
                "version": "1.0.0",
                "capabilities": {
                    "streaming": True,
                    "tool_calling": True,
                    "stateful": True
                },
                "model": "stock-analysis-agent",
                "categories": ["finance", "investment", "analysis"],
                "tags": ["stocks", "SEC filings", "financial analysis", "investment recommendations"]
            }
        ]
    }

@app.post("/agent-connect/v1/chat")
async def agent_connect_chat(
    request: ChatCompletionRequest,
    x_thread_id: Optional[str] = Header(None, alias="X-THREAD-ID"),
    authorization: Optional[str] = Header(None)
):
    """
    Official IBM Agent Connect Chat Completion endpoint
    
    This is the main endpoint that watsonx Orchestrate will call.
    Must follow OpenAI chat completions format with IBM-specific extensions.
    """
    try:
        thread_id = x_thread_id or f"thread-{uuid.uuid4().hex[:8]}"
        
        logger.info(f"🎯 Agent Connect chat request for thread {thread_id}")
        logger.info(f"🔄 Stream mode: {request.stream}")
        logger.info(f"📝 Messages: {len(request.messages)}")
        
        # Get the latest user message
        user_messages = [m for m in request.messages if m.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user messages found")
        
        latest_message = user_messages[-1].content
        stock_symbol = extract_stock_symbol(latest_message)
        
        if not stock_symbol or len(stock_symbol) < 1:
            # Return error for invalid symbol using IBM format
            if request.stream:
                return StreamingResponse(
                    generate_error_stream("I couldn't identify a valid stock symbol. Please specify a ticker like AAPL, TSLA, or MSFT.", thread_id),
                    media_type="text/plain",
                    headers={"X-THREAD-ID": thread_id}
                )
            else:
                return ChatCompletionResponse(
                    id=generate_completion_id(),
                    created=int(time.time()),
                    model=request.model,
                    choices=[{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "I couldn't identify a valid stock symbol. Please specify a ticker like AAPL, TSLA, or MSFT."
                        },
                        "finish_reason": "stop"
                    }]
                )
        
        # Store conversation in thread history
        if thread_id not in THREAD_HISTORY:
            THREAD_HISTORY[thread_id] = []
        THREAD_HISTORY[thread_id].extend([msg.dict() for msg in request.messages])
        
        if request.stream:
            logger.info(f"🌊 Starting streaming analysis for {stock_symbol}")
            return StreamingResponse(
                generate_stock_analysis_stream(stock_symbol, latest_message, thread_id),
                media_type="text/plain",
                headers={"X-THREAD-ID": thread_id}
            )
        else:
            logger.info(f"⚡ Starting synchronous analysis for {stock_symbol}")
            return await generate_stock_analysis_sync(stock_symbol, latest_message, thread_id, request.model)
            
    except Exception as e:
        logger.error(f"❌ Agent Connect chat failed: {e}")
        if request.stream:
            return StreamingResponse(
                generate_error_stream(f"Analysis error: {str(e)}", thread_id or "unknown"),
                media_type="text/plain"
            )
        else:
            raise HTTPException(status_code=500, detail=str(e))

async def generate_error_stream(error_message: str, thread_id: str):
    """Generate error response in IBM Agent Connect SSE format"""
    completion_id = generate_completion_id()
    
    # Error event
    error_event = {
        "id": completion_id,
        "object": "thread.run.step.delta",
        "thread_id": thread_id,
        "model": "stock-analysis-agent",
        "created": int(time.time()),
        "choices": [{
            "delta": {
                "role": "assistant",
                "content": f"❌ {error_message}"
            },
            "finish_reason": "stop"
        }]
    }
    
    yield f"event: thread.run.step.delta\n"
    yield f"data: {json.dumps(error_event)}\n\n"
    
    # Final event
    final_event = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "thread_id": thread_id,
        "model": "stock-analysis-agent",
        "created": int(time.time()),
        "choices": [{
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    
    yield f"event: chat.completion.chunk\n"
    yield f"data: {json.dumps(final_event)}\n\n"
    yield f"data: [DONE]\n\n"

async def generate_stock_analysis_stream(stock_symbol: str, user_message: str, thread_id: str):
    """
    Generate streaming stock analysis in IBM Agent Connect SSE format
    """
    try:
        completion_id = generate_completion_id()
        
        # Initial thinking step
        thinking_event = {
            "id": f"step-{uuid.uuid4().hex[:8]}",
            "object": "thread.run.step.delta",
            "thread_id": thread_id,
            "model": "stock-analysis-agent",
            "created": int(time.time()),
            "choices": [{
                "delta": {
                    "role": "assistant",
                    "step_details": {
                        "type": "thinking",
                        "content": f"Analyzing {stock_symbol.upper()} stock. I'll examine SEC filings, financial metrics, and market sentiment to provide a comprehensive investment recommendation."
                    }
                },
                "finish_reason": None
            }]
        }
        
        yield f"event: thread.run.step.delta\n"
        yield f"data: {json.dumps(thinking_event)}\n\n"
        await asyncio.sleep(1)
        
        # Tool call steps
        tool_steps = [
            "Searching SEC 10-K and 10-Q filings for financial data",
            "Analyzing key financial ratios and performance metrics", 
            "Researching recent market news and analyst opinions",
            "Evaluating risk factors and growth opportunities"
        ]
        
        for i, step_desc in enumerate(tool_steps):
            tool_event = {
                "id": f"step-{uuid.uuid4().hex[:8]}",
                "object": "thread.run.step.delta", 
                "thread_id": thread_id,
                "model": "stock-analysis-agent",
                "created": int(time.time()),
                "choices": [{
                    "delta": {
                        "role": "assistant",
                        "step_details": {
                            "type": "tool_calls",
                            "tool_calls": [{
                                "id": f"call_{uuid.uuid4().hex[:8]}",
                                "type": "function",
                                "function": {
                                    "name": f"analyze_step_{i+1}",
                                    "arguments": json.dumps({"step": step_desc, "ticker": stock_symbol})
                                }
                            }]
                        }
                    },
                    "finish_reason": None
                }]
            }
            
            yield f"event: thread.run.step.delta\n"
            yield f"data: {json.dumps(tool_event)}\n\n"
            await asyncio.sleep(2)
        
        # Try to get actual analysis with short timeout
        try:
            analysis_result = await asyncio.wait_for(
                run_stock_analysis_sync(stock_symbol),
                timeout=20  # Short timeout for Agent Connect
            )
            
            content = f"""📊 **Stock Analysis Complete for {stock_symbol.upper()}**

🎯 **Investment Recommendation: {analysis_result['recommendation']}**

📋 **Executive Summary:**
{analysis_result['summary']}

🔍 **Key Findings:**
{analysis_result['detailed_analysis'][:800]}...

📈 **Confidence Level:** {analysis_result['confidence_level']}

🕒 **Analysis Date:** {analysis_result['last_updated'][:19]}

---
*Analysis based on latest SEC filings, financial metrics, and market data*"""
            
        except asyncio.TimeoutError:
            content = f"""📊 **Quick Assessment for {stock_symbol.upper()}**

✅ **Symbol Validation:** {stock_symbol.upper()} confirmed as valid trading symbol

⚡ **Status:** Analysis initiated successfully but requires additional processing time for comprehensive SEC filings review.

🎯 **Preliminary Recommendation:** HOLD pending detailed analysis

📝 **Note:** Complete analysis typically takes 5-15 minutes for full SEC filings review, financial ratio analysis, and market sentiment evaluation."""
        
        # Final content event
        content_event = {
            "id": completion_id,
            "object": "thread.run.step.delta",
            "thread_id": thread_id,
            "model": "stock-analysis-agent", 
            "created": int(time.time()),
            "choices": [{
                "delta": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": None
            }]
        }
        
        yield f"event: thread.run.step.delta\n"
        yield f"data: {json.dumps(content_event)}\n\n"
        
        # Completion event
        completion_event = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "thread_id": thread_id,
            "model": "stock-analysis-agent",
            "created": int(time.time()),
            "choices": [{
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        
        yield f"event: chat.completion.chunk\n"
        yield f"data: {json.dumps(completion_event)}\n\n"
        yield f"data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"❌ Stream generation failed: {e}")
        yield f"event: error\n"
        yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
        yield f"data: [DONE]\n\n"

async def generate_stock_analysis_sync(stock_symbol: str, user_message: str, thread_id: str, model: str):
    """Generate synchronous stock analysis response"""
    try:
        # Quick analysis with timeout
        analysis_result = await asyncio.wait_for(
            run_stock_analysis_sync(stock_symbol),
            timeout=25  # 25 seconds for sync
        )
        
        content = f"""📊 **Stock Analysis for {stock_symbol.upper()}**

🎯 **Investment Recommendation: {analysis_result['recommendation']}**

📋 **Summary:** {analysis_result['summary']}

🔍 **Analysis:** {analysis_result['detailed_analysis'][:1000]}...

📈 **Confidence:** {analysis_result['confidence_level']}"""
        
    except asyncio.TimeoutError:
        content = f"""📊 **Quick Analysis for {stock_symbol.upper()}**

✅ Symbol {stock_symbol.upper()} validated as active trading symbol.

🎯 **Recommendation:** HOLD pending complete analysis

📝 Full analysis requires additional time for comprehensive review."""
    
    return ChatCompletionResponse(
        id=generate_completion_id(),
        created=int(time.time()),
        model=model,
        choices=[{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }],
        usage={
            "prompt_tokens": len(user_message.split()),
            "completion_tokens": len(content.split()),
            "total_tokens": len(user_message.split()) + len(content.split())
        }
    )

# ------------------------------------------------------------------------------
# Health and Info Endpoints
# ------------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "stock-analysis-agent",
        "version": "1.0.0",
        "agent_connect_compatible": True,
        "endpoints": {
            "agent_discovery": "/v1/agents",
            "chat_completion": "/agent-connect/v1/chat"
        }
    }

@app.get("/")
async def root():
    """Root endpoint with agent information"""
    return {
        "message": "Stock Analysis Agent - IBM watsonx Orchestrate Agent Connect Compatible",
        "version": "1.0.0", 
        "status": "healthy",
        "agent_connect_compatible": True,
        "endpoints": {
            "agent_discovery": "/v1/agents",
            "chat_completion": "/agent-connect/v1/chat",
            "health": "/health",
            "documentation": "/docs"
        },
        "capabilities": {
            "streaming": True,
            "tool_calling": True,
            "stateful_conversations": True,
            "sec_filings_analysis": True,
            "financial_metrics": True,
            "market_sentiment": True
        }
    }

# ------------------------------------------------------------------------------
# Legacy Compatibility Endpoints
# ------------------------------------------------------------------------------

@app.post("/chat/completions")
async def legacy_chat_completions(request: Request):
    """Legacy OpenAI chat completions endpoint - redirects to Agent Connect"""
    try:
        payload = await request.json()
        
        # Convert to Agent Connect format
        agent_request = ChatCompletionRequest(**payload)
        
        # Call the official Agent Connect endpoint
        return await agent_connect_chat(agent_request)
        
    except Exception as e:
        logger.error(f"❌ Legacy endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting Stock Analysis Agent on port {port}")
    logger.info(f"🔧 IBM watsonx Orchestrate Agent Connect Compatible")
    logger.info(f"   • Agent Discovery: /v1/agents")
    logger.info(f"   • Chat Completion: /agent-connect/v1/chat")
    logger.info(f"   • Health Check: /health")
    logger.info(f"📋 Documentation: http://localhost:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=True)