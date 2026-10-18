"""
兴义文旅 AI 导游 - 智能体核心模块
负责与智谱 GLM 大模型交互：
1. 通过 SYSTEM_PROMPT 设定「兴义文旅 AI 导游」角色与安全红线；
2. 以标准 OpenAI tools 参数 + tool_choice="auto" 注册知识库检索工具，
   由大模型自主决策何时调用（Function Calling），代码不做任何硬编码关键词匹配；
3. 拼接多轮历史对话，让回答具备上下文连贯性。
"""

import json
import logging
import os

from dotenv import load_dotenv
from zhipuai import ZhipuAI

import knowledge

# 加载 .env 中的 ZHIPU_API_KEY
load_dotenv()

logger = logging.getLogger("xingyi-tour")

# 初始化智谱 AI 客户端；未配置 Key 时为 None，调用时抛出异常由上层兜底
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
client = ZhipuAI(api_key=ZHIPU_API_KEY) if ZHIPU_API_KEY else None
GLM_MODEL = "glm-4-flash"  # 智谱免费模型

# ---------------------------------------------------------------------------
# 系统提示词：角色设定 + 安全红线 + 回答要求
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是"兴义文旅 AI 导游"，一位热情专业的贵州兴义本地导游，只为游客解答与兴义旅游相关的问题。

【职责范围】
1. 只回答与贵州兴义市旅游相关的问题，包括景点、美食、酒店住宿、游玩路线、交通、活动体验、民俗文化等；
2. 只要用户的问题与兴义旅游相关（无论是问门票、地址、开放时间、价格，还是问"有什么好吃的/好玩的/怎么安排"等推荐类问题），都必须先调用 search_knowledge 工具检索知识库，再依据检索结果作答，不得仅凭自身知识直接回答；
3. 当用户询问旅游规划或推荐，但未说明游玩天数、人员构成或兴趣偏好时，你必须先向用户提问以获取完整信息，不要直接瞎编行程；
4. 回答口吻亲切自然，像真人导游一样。

【安全红线】
1. 如果用户询问与旅游无关的问题（例如写代码、游戏外挂、政治、新闻时事等），或涉及敏感、违规的内容，必须礼貌拒绝，并强行把话题引导回兴义旅游。示例话术："抱歉，这个问题我帮不了您～不过兴义的万峰林、马岭河峡谷都很值得一游，需要我介绍一下吗？";
2. 绝不回答任何违法违规、色情、暴力、政治敏感内容，绝不协助任何作弊、外挂、攻击行为。

【回答要求】
回答简明扼要，控制在 200 字以内，重点突出。"""

# ---------------------------------------------------------------------------
# Function Calling 工具注册表（标准 OpenAI tools 格式）
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "检索兴义本地文旅知识库，获取景点、美食、酒店、游玩路线、交通、"
                "活动体验、民俗文化、游客评价等准确资料。当用户询问兴义旅游的"
                "具体信息（如门票、地址、开放时间、美食推荐、路线规划）时必须调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索关键词，例如「万峰林门票」「兴义美食」「马岭河峡谷」",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


def chat(question: str, history: list[dict] | None = None) -> dict:
    """
    与智谱 GLM 大模型完成一轮对话，支持 Function Calling 自主检索知识库。

    输入：
        question (str): 用户本轮提问文本
        history (list[dict] | None): 历史对话列表，元素为
            {"role": "user"/"assistant", "content": "..."}，仅取最近约 6 轮（12 条）
    输出：
        dict: {
            "reply": str,             # 大模型生成的回答文本
            "sources": list[dict],    # 本轮实际检索到的知识来源（含来源与更新时间）
            "ai": bool                # 是否由大模型生成（成功路径恒为 True）
        }
    异常：
        大模型 API 超时或失败时向上抛出，由接口层统一捕获并返回友好提示。
    """
    if client is None:
        raise RuntimeError("ZHIPU_API_KEY 未配置，无法调用大模型")

    # 本轮问答中被检索命中的知识来源（去重收集，供前端展示）
    collected_sources: list[dict] = []

    def run_search_knowledge(query: str) -> str:
        """
        search_knowledge 工具的本地实现：检索知识库并把结果格式化后交还给大模型。

        输入：query (str) - 大模型给出的检索关键词
        输出：str - 格式化后的知识库资料文本，供大模型阅读后生成回答
        副作用：把命中条目的来源信息追加到 collected_sources
        """
        logger.info(f"大模型自主调用 search_knowledge，关键词: {query}")
        matches = knowledge.search(query, limit=3)
        for m in matches:
            info = knowledge.get_source_info(m)
            if all(s["id"] != info["id"] for s in collected_sources):
                collected_sources.append(info)
        if not matches:
            return "知识库中未检索到相关资料。"
        return "\n\n".join(knowledge.format_answer(m) for m in matches)

    # 组装消息：系统提示词 + 最近约 6 轮历史 + 本轮提问
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in (history or [])[-12:]:
        role = h.get("role")
        content = h.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})

    # Function Calling 循环：大模型可能连续多轮调用工具，最多允许 3 轮防止死循环
    for round_idx in range(3):
        resp = client.chat.completions.create(
            model=GLM_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",  # 由大模型自主决定是否调用工具
            temperature=0.7,
            max_tokens=600,
        )

        # 大模型输出格式解析兜底：解析异常时降级为普通文本回复，绝不抛出 500
        try:
            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)

            # 大模型决定直接回答，不再调用工具 -> 返回最终结果
            if not tool_calls:
                reply = (msg.content or "").strip()
                logger.info(
                    f"GLM 生成成功（第 {round_idx + 1} 轮），回答长度 {len(reply)} 字，"
                    f"tokens: {resp.usage.total_tokens}"
                )
                return {"reply": reply, "sources": collected_sources, "ai": True}

            # 大模型决定调用工具：先把 assistant 的工具调用消息加入上下文
            assistant_msg = {"role": "assistant", "content": msg.content or ""}
            assistant_msg["tool_calls"] = [
                tc.model_dump() if hasattr(tc, "model_dump") else tc for tc in tool_calls
            ]
            messages.append(assistant_msg)

            # 逐个执行工具调用，把结果以 role="tool" 回传给大模型
            for tc in tool_calls:
                if tc.function.name == "search_knowledge":
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        # 工具参数 JSON 解析失败：用用户原始问题兜底检索
                        logger.warning(f"工具参数解析失败，使用原始问题兜底: {tc.function.arguments!r}")
                        args = {}
                    result = run_search_knowledge(args.get("query") or question)
                else:
                    result = "未知工具，无法执行。"
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )
        except (json.JSONDecodeError, AttributeError, IndexError, TypeError) as e:
            # 输出结构不符合预期（缺 choices / message 异常等）：尝试取文本内容，取不到则给兜底文案
            logger.error(f"大模型输出格式解析失败（第 {round_idx + 1} 轮）: {type(e).__name__}: {e}")
            fallback_text = ""
            try:
                fallback_text = (resp.choices[0].message.content or "").strip()
            except (AttributeError, IndexError, TypeError):
                pass
            return {
                "reply": fallback_text or "系统开小差了，请再说一遍。",
                "sources": collected_sources,
                "ai": True,
            }

    # 理论上不会走到这里（3 轮内大模型应给出最终回答），兜底返回
    logger.warning("Function Calling 超过最大轮次，返回兜底提示")
    return {
        "reply": "这个问题有点复杂，让我想想～您可以换个方式问我，比如「万峰林门票多少钱」「兴义有什么美食」。",
        "sources": collected_sources,
        "ai": True,
    }
