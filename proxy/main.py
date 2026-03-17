from fastapi import FastAPI, Request
import requests
from datetime import datetime

app = FastAPI()

OLLAMA_URL_CHAT = "http://ollama:11434/api/chat"
OLLAMA_URL_GENERATE = "http://ollama:11434/api/generate"

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "llama3:8b")
    messages = body.get("messages", [])

    ollama_payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": 128}
    }

    resp = requests.post(OLLAMA_URL_CHAT, json=ollama_payload, timeout=120)
    data = resp.json()
    content = data.get("message", {}).get("content", "")

    return {
        "id": "chatcmpl-ollama",
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(str(messages)),
            "completion_tokens": len(content.split()),
            "total_tokens": len(str(messages)) + len(content.split())
        }
    }

@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    model = body.get("model", "llama3:8b")
    prompt = body.get("prompt", "")

    ollama_payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 128}
    }

    resp = requests.post(OLLAMA_URL_GENERATE, json=ollama_payload, timeout=120)
    data = resp.json()
    content = data.get("response", "")

    return {
        "id": "cmpl-ollama",
        "object": "text_completion",
        "created": int(datetime.now().timestamp()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "text": content,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(content.split()),
            "total_tokens": len(prompt.split()) + len(content.split())
        }
    }
