# main.py
from fastapi import FastAPI, APIRouter, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import time
import uuid
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# ====== 数据模型 ======
class ChatMessage(BaseModel):
    role: str  # "system"|"user"|"assistant"
    content: str


class ChatRequest(BaseModel):
    model: str = "gpt-3.5-turbo"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7


class ChatResponseChoice(BaseModel):
    index: int
    message: ChatMessage


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatResponseChoice]


# ====== 服务类 ======
class OpenAIService:
    async def generate_response(self, request: ChatRequest):
        # 固定响应实现（兼容OpenAI格式）
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "# 我是小智障,马上就变成大聪明了,所以我现在一句话不会回答你哦😀"}}],
        }


# ====== 依赖项 ======
api_key_header = APIKeyHeader(name="Authorization")


async def verify_api_key(api_key: str = Depends(api_key_header)):
    expected_key = "123456"  # 直接在这里设置你的API密钥
    if not api_key.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication format")
    if api_key[7:] != expected_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    return True


# ====== FastAPI实例 ======
app = FastAPI(title="OpenAI Compatible API", description="用于Open WebUI的兼容接口", version="0.1.0")

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====== 路由端点 ======
@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completion(request: ChatRequest, _: bool = Depends(verify_api_key), service: OpenAIService = Depends(OpenAIService)):
    print(request)
    return await service.generate_response(request)


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": "小智障", "object": "model", "created": 1686935002, "owned_by": "openai"}]}


@app.get("/health")
def health_check():
    return {"status": "healthy"}
