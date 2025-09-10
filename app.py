# app.py - Final watsonx Orchestrate Compatible Version
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
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
# FastAPI app with watsonx Orchestrate compatibility
# ------------------------------------------------------------------------------
app = FastAPI(
    title="Stock Analysis API",
    description="AI-powered stock analysis using CrewAI with watsonx Orchestrate integration",
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
# In-memory cache
# ------------------------------------------------------------------------------
RESULT_CACHE: Dict[str, Dict[str, Any]] = {}

# ------------------------------------------------------------------------------
# watsonx Orchestrate Models (Simplified for skill-based approach)
# ------------------------------------------------------------------------------
class StockAnalysisInput(BaseModel):
    """Input for stock analysis skill"""
    stock_symbol: str = Field(..., description="Stock ticker symbol (e.g., AAPL, TSLA, MSFT)")
    analysis_type: Optional[str] = Field("comprehensive", description="Type of analysis: comprehensive, quick, or detailed")

class StockAnalysisOutput(BaseModel):
    """Output for stock analysis skill"""
    stock_symbol: str = Field(..., description="Analyzed stock symbol")
    recommendation: str = Field(..., description="Investment recommendation (Buy/Hold/Sell)")
    summary: str = Field(..., description="Executive summary of the analysis")
    detailed_analysis: str = Field(..., description="Full detailed analysis")
    confidence_level: str = Field(..., description="Confidence level (High/Medium/Low)")
    last_updated: str = Field(..., description="When the analysis was completed")

class HealthCheckOutput(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")

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
# watsonx Orchestrate Skill Endpoints
# ------------------------------------------------------------------------------

@app.post("/analyze-stock", response_model=StockAnalysisOutput, 
          summary="Analyze Stock Performance",
          description="Provides comprehensive stock analysis and investment recommendations using AI agents")
async def analyze_stock_skill(input_data: StockAnalysisInput) -> StockAnalysisOutput:
    """
    watsonx Orchestrate Skill: Analyze a stock and provide investment recommendation
    
    This endpoint is designed specifically for watsonx Orchestrate integration.
    It takes a stock symbol and returns a structured analysis with recommendation.
    """
    try:
        # Extract and validate stock symbol
        stock_symbol = extract_stock_symbol(input_data.stock_symbol)
        
        if not stock_symbol or len(stock_symbol) < 1:
            raise HTTPException(status_code=400, detail="Invalid stock symbol provided")
        
        logger.info(f"🎯 watsonx Orchestrate skill called for {stock_symbol}")
        
        # Run analysis
        result = await run_stock_analysis_sync(stock_symbol)
        
        # Return structured response for watsonx Orchestrate
        return StockAnalysisOutput(**result)
        
    except Exception as e:
        logger.error(f"❌ Skill execution failed: {e}")
        # Return error as valid response for watsonx Orchestrate
        return StockAnalysisOutput(
            stock_symbol=input_data.stock_symbol.upper(),
            recommendation="HOLD",
            summary=f"Error analyzing {input_data.stock_symbol}: {str(e)}",
            detailed_analysis=f"Analysis failed due to: {str(e)}. Please verify the stock symbol and try again.",
            confidence_level="Low",
            last_updated=datetime.now().isoformat()
        )

@app.get("/health", response_model=HealthCheckOutput,
         summary="Health Check",
         description="Check if the stock analysis service is running properly")
async def health_check() -> HealthCheckOutput:
    """
    Health check endpoint for watsonx Orchestrate
    """
    return HealthCheckOutput(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

# ------------------------------------------------------------------------------
# Legacy endpoints (for backward compatibility)
# ------------------------------------------------------------------------------

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Stock Analysis API - watsonx Orchestrate Compatible",
        "version": "1.0.0",
        "status": "healthy",
        "watsonx_compatible": True,
        "skills": [
            {
                "name": "analyze-stock",
                "description": "Analyze stock performance and provide investment recommendations",
                "endpoint": "/analyze-stock"
            }
        ],
        "openapi_spec": "/docs",
        "endpoints": {
            "watsonx_skill": "/analyze-stock",
            "health": "/health",
            "documentation": "/docs"
        }
    }

@app.post("/chat/completions")
async def chat_completions_legacy(request: Request):
    """
    Legacy chat completions endpoint - redirects to skill-based approach
    """
    try:
        payload = await request.json()
        messages = payload.get("messages", [])
        
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        # Extract stock symbol from the last user message
        user_message = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        stock_symbol = extract_stock_symbol(user_message)
        
        # Call the skill endpoint
        skill_input = StockAnalysisInput(stock_symbol=stock_symbol)
        result = await analyze_stock_skill(skill_input)
        
        # Convert to chat completion format
        content = f"**Stock Analysis for {result.stock_symbol}**\n\n"
        content += f"**Recommendation:** {result.recommendation}\n\n"
        content += f"**Summary:** {result.summary}\n\n"
        content += f"**Confidence:** {result.confidence_level}\n\n"
        content += f"**Detailed Analysis:**\n{result.detailed_analysis}"
        
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
            "model": "stock-analysis-agent",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(content.split()),
                "total_tokens": len(user_message.split()) + len(content.split())
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting Stock Analysis API on port {port}")
    logger.info(f"🔧 watsonx Orchestrate Skills Available:")
    logger.info(f"   • /analyze-stock - Stock analysis skill")
    logger.info(f"   • /health - Health check")
    logger.info(f"📋 OpenAPI Documentation: http://localhost:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=True)