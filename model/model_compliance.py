import os
import time
import uuid
import json
from typing import List, Optional
from threading import Thread

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# ====== 数据模型 ======
class ChatMessage(BaseModel):
    role: str  # "system"|"user"|"assistant"
    content: str


class ChatRequest(BaseModel):
    model: str = "FinSynth"
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


tokenizer = AutoTokenizer.from_pretrained(
    "Fintech-Dreamer/FinSynth_model_compliance",
    trust_remote_code=True,
    padding_side="left",
    truncation_side="left",
)

print("正在加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    "Fintech-Dreamer/FinSynth_model_compliance",
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    use_cache=True,
).eval()


# ====== 服务类 ======
class OpenAIService:
    async def generate_response(self, request: ChatRequest):
        prompt = request.messages[-1].content
        try:
            # 构建完整提示词
            full_prompt = f'Assistant:["是","否"]\nUser: {prompt}\nAssistant:'
            inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, padding=True, add_special_tokens=True).to(model.device)

            # 创建流式处理器
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            # 启动生成线程
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=5000,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )

            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            # 实时流式输出
            for token in streamer:
                print(token)
                yield f"data: {json.dumps({'id': f'chatcmpl-{uuid.uuid4()}', 'object': 'chat.completion', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': token}}]})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'id': f'chatcmpl-{uuid.uuid4()}', 'object': 'chat.completion', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': f'[ERROR] {str(e)}'}}]})}\n\n"


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


# ====== 自定义异常处理 ======
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        content={
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": None,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "很抱歉以我的能力无法处理上面信息,我会努力改进!"}}],
        },
        status_code=200,
    )


# ====== 路由端点 ======
@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completion(request: ChatRequest, _: bool = Depends(verify_api_key), service: OpenAIService = Depends(OpenAIService)):
    return StreamingResponse(service.generate_response(request), media_type="text/event-stream")


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "FinSynth-compliance",
                "name": "FinSynth-compliance",
                "meta": {
                    "profile_image_url": "/favicon.png",
                    "description": "合规监控机器人",
                    "model_ids": None,
                },
            }
        ],
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}
