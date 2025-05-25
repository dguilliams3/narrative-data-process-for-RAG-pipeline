# gpt_service.py
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from logging_utils import configure_logging, ensure_log_dir

# Setup logging
LOG_FILE_PATH = "logs/gpt_service.log"
ensure_log_dir(LOG_FILE_PATH)
configure_logging(LOG_FILE_PATH)
logger = logging.getLogger(__name__)

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    model: str
    max_tokens: int
    temperature: float
    api_key: str

# HTTP exceptions function for cleaner service
def validate_gpt_service_call(req: GenerateRequest, api_key: str):
    """Validate the request parameters for the GPT service call."""
    if not api_key:
        logger.error("OPENAI_API_KEY missing")
        raise HTTPException(status_code=500, detail="API key not configured")
    if req.prompt is None or req.prompt.strip() == "":
        logger.error("Prompt is empty")
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    if req.model is None or req.model.strip() == "":
        logger.error("Model is empty")
        raise HTTPException(status_code=400, detail="Model cannot be empty")
    if req.max_tokens <= 0:
        logger.error("Max tokens must be greater than 0")
        raise HTTPException(status_code=400, detail="Max tokens must be greater than 0")
    if req.temperature < 0 or req.temperature > 1:
        logger.error("Temperature must be between 0 and 1")
        raise HTTPException(status_code=400, detail="Temperature must be between 0 and 1")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate")
def generate(req: GenerateRequest):
    # Validate the request parameters.  We enforce the API key being sent here 
    api_key = req.api_key
    validate_gpt_service_call(req, api_key)
    client = OpenAI(api_key=api_key)

    try:
        resp = client.chat.completions.create(
            model=req.model,
            messages=[{"role": "user", "content": req.prompt}],
            max_tokens=req.max_tokens,
            temperature=req.temperature
        )
        content = resp.choices[0].message.content.strip()
        logger.info("GPT response length=%d", len(content))
        return {"content": content}
    except Exception:
        logger.exception("OpenAI call failed")
        raise HTTPException(status_code=502, detail="LLM request failed")