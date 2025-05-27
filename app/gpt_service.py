# gpt_service.py
import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from gpt_config import SUMMARIZER_PROMPT_TEMPLATE
from logging_utils import configure_logging, ensure_log_dir
import time

# Setup logging
LOG_FILE_PATH = "logs/gpt_service.log"
ensure_log_dir(LOG_FILE_PATH)
configure_logging(LOG_FILE_PATH)
logger = logging.getLogger(__name__)

app = FastAPI()
tokens_used = 0
request_count = 0
start_time = time.time()

class GPTRequest(BaseModel):
    prompt: str
    api_key: str
    model: str
    max_tokens: int
    temperature: float

class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class GPTResponse(BaseModel):
    content: str
    model: str
    usage: TokenUsage
    finish_reason: str
    duration: float

def _send_to_openai(
    prompt: str,
    api_key: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, dict]:
    """
    Internal function that handles the direct OpenAI API call.
    All params are explicit to make it clear what's being sent.
    """
    client = OpenAI(api_key=api_key)
    response_metadata = {}
    
    try:
        start = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        duration = time.time() - start
        content = resp.choices[0].message.content.strip()
        
        # Add to the total tokens used counter
        global tokens_used
        tokens_used += resp.usage.total_tokens

        # Assign metadata, we'll use this with metrics later
        response_metadata = {
            "duration": duration,
            "model": model,
            "usage": {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens
            },
            "finish_reason": resp.choices[0].finish_reason
        }

        logger.info(
            "[OPENAI CALL DURATION] %.2fs",
            duration
        )
        logger.info(
            "Full response: %s",
            content
        )

        return content, response_metadata

    except Exception as e:
        logger.exception("OpenAI API call failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate", response_model=GPTResponse)
async def generate(req: GPTRequest):
    """
    Generate text using OpenAI's GPT models.

    This endpoint provides a clean interface to OpenAI's GPT models, handling all the API interaction
    and error handling. The client is responsible for prompt formatting and context management.

    Parameters:
    - **prompt**: The formatted prompt to send to GPT
    - **api_key**: OpenAI API key for authentication
    - **model**: The GPT model to use (e.g., "gpt-4", "gpt-3.5-turbo")
    - **max_tokens**: Maximum number of tokens to generate
    - **temperature**: Controls randomness (0.0-1.0, lower is more deterministic)

    Returns:
    - **content**: The generated text response
    - **model**: The model used for generation
    - **usage**: Token usage statistics (prompt_tokens, completion_tokens, total_tokens)
    - **finish_reason**: Why the model stopped generating (e.g., "stop", "length", "content_filter")
    - **duration**: How long the request took in seconds

    Raises:
    - **HTTPException(500)**: If the OpenAI API call fails
    """
    global request_count
    request_count += 1

    try:
        content, metadata = _send_to_openai(
            prompt=req.prompt,
            api_key=req.api_key,
            model=req.model,
            max_tokens=req.max_tokens,
            temperature=req.temperature
        )
        return GPTResponse(
            content=content,
            model=metadata["model"],
            usage=TokenUsage(**metadata["usage"]),
            finish_reason=metadata["finish_reason"],
            duration=metadata["duration"]
        )
    except Exception as e:
        logger.exception("Failed to generate response")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    """
    Check the health status of the GPT service.

    This endpoint can be used by monitoring tools and load balancers to verify
    that the service is up and running.

    Returns:
    - **status**: "ok" if the service is healthy
    - **uptime**: The uptime of the service in seconds
    - **request_count**: The number of requests the service has processed
    """
    uptime = time.time() - start_time
    return {"status": "ok", "uptime": uptime, "request_count": request_count}

@app.get("/metrics")
def metrics():
    """
    Provide metrics about the GPT service.

    This endpoint returns a JSON object with the total tokens used and the
    average tokens per request.
    """
    return {"total_tokens": tokens_used, "avg_tokens_per_request": tokens_used / request_count}