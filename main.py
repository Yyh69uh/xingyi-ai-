"""
兴义文旅 AI 导游 - 后端服务
基于 FastAPI 框架，整合兴义文旅知识库与智谱 GLM 大模型，
提供 AI 导游问答（Function Calling 自主检索）与知识库数据浏览接口。
大模型交互逻辑全部封装在 agent_core 模块，本文件只负责路由与统一容错。
"""

import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import agent_core
import knowledge

# 启动时加载 .env 文件中的环境变量
load_dotenv()

# 配置日志格式，让运行过程清晰可见
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("xingyi-tour")

app = FastAPI(title="兴义文旅 AI 导游", version="2.0.0")

# 允许跨域，方便前端 fetch 调用后端接口
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent


class ChatRequest(BaseModel):
    """问答接口请求体：question 为本轮提问，history 为前端携带的历史对话"""

    question: str
    history: list[dict] = []


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    全局入参校验异常处理（非法输入拦截）。

    输入：校验失败的请求（如字段类型错误、JSON 结构非法）
    输出：JSONResponse - 友好提示，HTTP 200，绝不向前端暴露 422 校验细节
    作用：防止畸形请求体导致前端看到红色报错。
    """
    logger.warning(f"非法请求被拦截: {request.url.path} - {exc.errors()[:2]}")
    return JSONResponse(
        status_code=200,
        content={"reply": "输入格式有误，请重新提问哦。", "sources": [], "ai": False},
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """中间件：记录每个请求的方法、路径、耗时和状态码"""
    start_time = time.time()
    logger.info(f"收到请求: {request.method} {request.url.path}")
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    logger.info(
        f"响应完成: {request.method} {request.url.path} "
        f"-> {response.status_code} (耗时 {process_time:.1f}ms)"
    )
    return response


@app.on_event("startup")
async def startup_event():
    """
    服务启动事件：加载知识库并检查环境变量。

    输入/输出：无
    作用：启动时一次性把 knowledge/ 目录数据加载到内存，并确认大模型 Key 已配置。
    """
    logger.info("=" * 60)
    logger.info("兴义文旅 AI 导游 后端服务正在启动...")
    counts = knowledge.load_knowledge()
    for cat, n in counts.items():
        logger.info(f"  - {cat}: {n} 条")
    # 检查环境变量加载状态（不显示密钥任何字符）
    if os.getenv("ZHIPU_API_KEY"):
        logger.info("环境变量 ZHIPU_API_KEY 已成功加载")
    else:
        logger.warning("环境变量 ZHIPU_API_KEY 未配置")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭事件"""
    logger.info("兴义文旅 AI 导游 后端服务已停止。")


@app.get("/api/health")
def health_check():
    """
    健康检查接口。

    输入：无
    输出：dict - {"status": "ok", "message": 提示文本}
    作用：供前端页面加载时确认后端连通性。
    """
    return {"status": "ok", "message": "兴义文旅 AI 导游后端已启动！"}


@app.post("/api/chat")
def chat(req: ChatRequest):
    """
    AI 导游问答接口（大模型 + Function Calling 自主检索知识库）。

    输入：ChatRequest - {"question": 用户提问, "history": 历史对话列表}
    输出：dict - {"reply": 回答文本, "sources": [来源信息...], "ai": 是否大模型生成}
    作用：把问答交给 agent_core 处理，由大模型自主决策是否检索知识库；
          外层统一 try...except 容错——大模型超时或失败时返回友好提示，
          严禁向前端抛出堆栈信息。
    """
    question = req.question.strip()
    logger.info(f"收到问答请求: {question[:50]}（历史 {len(req.history)} 条）")

    # 异常类型1：空输入
    if not question:
        return {
            "reply": "请输入您想了解的问题，例如：万峰林门票多少钱？兴义有什么美食？",
            "sources": [],
            "ai": False,
        }

    # 异常类型2：超长输入拦截（防止超长文本压垮大模型调用）
    if len(question) > 500:
        logger.warning(f"超长输入被拦截，长度 {len(question)} 字符")
        return {
            "reply": "您输入的问题太长了，请精简后重新提问哦。",
            "sources": [],
            "ai": False,
        }

    # 异常类型3：非法 history 元素过滤（只保留合法的 user/assistant 对话）
    safe_history = [
        h for h in req.history
        if isinstance(h, dict)
        and h.get("role") in ("user", "assistant")
        and isinstance(h.get("content"), str)
    ]

    try:
        return agent_core.chat(question, safe_history)
    except json.JSONDecodeError as e:
        # 异常类型4：大模型输出 JSON 解析失败
        logger.error(f"大模型输出解析失败: {e}")
        return {"reply": "系统开小差了，请再说一遍。", "sources": [], "ai": False}
    except Exception as e:
        # 异常类型5：大模型 API 超时/网络故障等未知异常
        logger.error(f"大模型调用失败: {type(e).__name__}: {e}")
        return {"reply": "AI 导游目前网络拥堵，请稍后再试。", "sources": [], "ai": False}


@app.get("/api/knowledge/categories")
def get_categories():
    """
    知识库分类浏览接口。

    输入：无
    输出：dict - {"categories": [{"key", "name", "count"}, ...]}
    作用：返回知识库所有分类及条目数。
    """
    return {"categories": knowledge.list_categories()}


@app.get("/api/knowledge/{category}")
def get_category_items(category: str):
    """
    分类条目列表接口。

    输入：category (str) - 路径参数，分类目录名
    输出：dict - {"category": 分类名, "items": [简要条目...]}
    作用：返回某分类下所有条目的简要信息。
    """
    return {"category": category, "items": knowledge.list_items(category)}


@app.get("/api/knowledge/item/{item_id}")
def get_knowledge_item(item_id: str):
    """
    条目详情接口。

    输入：item_id (str) - 路径参数，条目唯一标识
    输出：dict - 完整知识条目；不存在时返回 {"error": "条目不存在"}
    作用：按 id 返回完整知识条目。
    """
    item = knowledge.get_item(item_id)
    if item is None:
        return {"error": "条目不存在"}
    return item


@app.get("/")
def index():
    """
    前端首页托管接口。

    输入：无
    输出：FileResponse - index.html 页面
    作用：在 8000 端口同时托管前端页面与后端接口。
    """
    return FileResponse(BASE_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    # 本地运行固定 8000 端口；云端平台（Render 等）会注入 PORT 环境变量，需优先采用
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False, log_level="info")
