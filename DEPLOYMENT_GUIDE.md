# Stock Analysis API - Render Deployment Guide

## Quick Start

### 1. Prepare Your Repository

1. Add all the new files to your existing repository:
   - `app.py` (FastAPI wrapper)
   - `requirements.txt` (Python dependencies)
   - `Dockerfile` (for containerized deployment)
   - `render.yaml` (Render configuration)
   - `start.sh` (startup script)

2. Update your project structure:
```
stock-analysis/
├── src/
│   └── stock_analysis/
├── app.py                 # New FastAPI wrapper
├── requirements.txt       # New requirements file
├── Dockerfile            # New Docker configuration
├── render.yaml           # New Render configuration
├── start.sh              # New startup script
├── .env.example          # Updated with FastAPI vars
└── README.md
```

### 2. Deploy to Render

#### Option A: Using Render Dashboard (Recommended)

1. **Sign up/Login to Render**: Go to [render.com](https://render.com)

2. **Connect Your Repository**: 
   - Click "New +" → "Web Service"
   - Connect your GitHub/GitLab repository
   - Select your stock analysis repository

3. **Configure Service**:
   - **Name**: `stock-analysis-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Plan**: Choose based on your needs (Starter plan works for testing)

4. **Set Environment Variables**:
   ```
   OPENAI_API_KEY=your_openai_key_here
   SEC_API_API_KEY=your_sec_api_key_here
   SERPER_API_KEY=your_serper_key_here (optional)
   BROWSERLESS_API_KEY=your_browserless_key_here (optional)
   ```

5. **Deploy**: Click "Create Web Service"

#### Option B: Using Render CLI

```bash
# Install Render CLI
npm install -g @render/cli

# Login to Render
render login

# Deploy using render.yaml
render deploy
```

### 3. Test Your Deployment

Once deployed, your API will be available at `https://your-service-name.onrender.com`

#### Test Endpoints:

1. **Health Check**:
```bash
curl https://your-service-name.onrender.com/health
```

2. **Start Analysis**:
```bash
curl -X POST "https://your-service-name.onrender.com/analyze" \
  -H "Content-Type: application/json" \
  -d '{"company_stock": "AAPL", "query": "Analyze Apple stock"}'
```

3. **Check Status**:
```bash
curl https://your-service-name.onrender.com/status/JOB_ID_HERE
```

## Watson X Orchestra Integration

### API Endpoints for Watson X Orchestra

Your deployed API provides these endpoints for integration:

- **POST /analyze**: Start asynchronous stock analysis
- **GET /status/{job_id}**: Check analysis status
- **GET /health**: Health check endpoint
- **POST /analyze/sync**: Synchronous analysis (use carefully)

### Example Integration Code

```python
import requests
import time

class StockAnalysisClient:
    def __init__(self, base_url):
        self.base_url = base_url.rstrip('/')
    
    def analyze_stock(self, company_stock, query=None):
        """Start stock analysis"""
        data = {"company_stock": company_stock}
        if query:
            data["query"] = query
            
        response = requests.post(f"{self.base_url}/analyze", json=data)
        return response.json()
    
    def get_status(self, job_id):
        """Get analysis status"""
        response = requests.get(f"{self.base_url}/status/{job_id}")
        return response.json()
    
    def wait_for_completion(self, job_id, timeout=300):
        """Wait for analysis to complete"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_status(job_id)
            if status["status"] in ["completed", "failed"]:
                return status
            time.sleep(10)
        return {"status": "timeout"}

# Usage
client = StockAnalysisClient("https://your-service-name.onrender.com")
result = client.analyze_stock("AAPL")
job_id = result["job_id"]
final_result = client.wait_for_completion(job_id)
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for LLM |
| `SEC_API_API_KEY` | Yes | SEC API key for filings |
| `SERPER_API_KEY` | No | Google Search API key |
| `BROWSERLESS_API_KEY` | No | Web scraping API key |
| `PORT` | No | Server port (default: 8000) |

### Scaling Considerations

1. **Render Plans**: 
   - Starter: Good for testing, may have timeouts for long analyses
   - Standard/Pro: Better for production workloads

2. **Background Jobs**: 
   - Current implementation uses in-memory storage
   - For production, consider using Redis or database

3. **Rate Limiting**: 
   - Implement rate limiting for production use
   - Consider API key-based authentication

## Troubleshooting

### Common Issues

1. **Build Failures**:
   - Check requirements.txt for version conflicts
   - Ensure all dependencies are compatible

2. **Runtime Errors**:
   - Verify environment variables are set
   - Check logs in Render dashboard

3. **Timeouts**:
   - Stock analysis can take 5-15 minutes
   - Use async endpoints for long-running tasks
   - Consider upgrading Render plan

4. **Memory Issues**:
   - CrewAI can be memory-intensive
   - Monitor memory usage in Render dashboard
   - Consider upgrading to higher memory plan

### Debugging

1. **Check Render Logs**:
   - Go to Render dashboard → Your service → Logs

2. **Test Locally**:
```bash
pip install -r requirements.txt
python app.py
# Test at http://localhost:8000
```

3. **Health Monitoring**:
   - Use `/health` endpoint for monitoring
   - Set up alerts in Render dashboard

## Security Considerations

1. **API Keys**: Store sensitive keys in Render environment variables
2. **CORS**: Configure proper CORS origins for production
3. **Rate Limiting**: Implement rate limiting for production APIs
4. **Authentication**: Consider adding API key authentication

## Cost Optimization

1. **Render Plans**: Start with Starter plan, upgrade as needed
2. **Sleep Behavior**: Render services sleep after inactivity on free plans
3. **Monitoring**: Track usage and costs in Render dashboard

## Support

- **Render Documentation**: [render.com/docs](https://render.com/docs)
- **CrewAI Documentation**: [docs.crewai.com](https://docs.crewai.com)
- **FastAPI Documentation**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com)