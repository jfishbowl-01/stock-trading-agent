from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging
import os
from datetime import datetime
import uuid

# Import your crew
from src.stock_analysis.crew import StockAnalysisCrew

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stock Analysis API",
    description="AI-powered stock analysis using CrewAI",
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

async def run_stock_analysis(job_id: str, company_stock: str, query: str):
    """Background task to run stock analysis"""
    try:
        logger.info(f"Starting analysis for {company_stock} (Job: {job_id})")
        job_status[job_id]["status"] = "running"
        
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

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Stock Analysis API is running",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Detailed health check for monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "stock-analysis-api",
        "version": "1.0.0"
    }

@app.post("/analyze", response_model=StockAnalysisResponse)
async def analyze_stock(
    request: StockAnalysisRequest, 
    background_tasks: BackgroundTasks
):
    """
    Start stock analysis for a given company
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
    """
    Get the status of a stock analysis job
    """
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatusResponse(**job_status[job_id])

@app.get("/jobs")
async def list_jobs():
    """
    List all jobs (for debugging/monitoring)
    """
    return {
        "total_jobs": len(job_status),
        "jobs": list(job_status.values())
    }

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job from memory
    """
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