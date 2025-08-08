
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
import logging
import uuid
import time
import json

from crew import crew
from utils import generate_stream

app = FastAPI()

logging.basicConfig(level=logging.INFO)

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.post("/chat/completions")
async def chat_completions(request: Request):
    try:
        req_data = await request.json()

        # 🛠 Handle Orchestrate input format (Watsonx)
        if "properties" in req_data:
            props = json.loads(req_data["properties"])
            stock = props.get("stock_ticker", "IBM")
            messages = [{"role": "user", "content": f"Should I invest in {stock} stock right now?"}]
            stream = False
        else:
            # Handle normal OpenAI-style input
            messages = req_data["messages"]
            stream = req_data.get("stream", False)

        # ✅ Log the request message
        logging.info(f"Received stock analysis request: {messages[0]['content']}")

        result = await crew.run(messages[0]["content"])

        if stream:
            return StreamingResponse(generate_stream(result), media_type="text/event-stream")
        else:
            return {
                "id": str(uuid.uuid4()),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "stock-analysis-agent",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": result},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(messages),
                    "completion_tokens": len(result),
                    "total_tokens": len(messages) + len(result),
                },
            }

    except Exception as e:
        logging.error(f"❌ Watson X chat completions failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
