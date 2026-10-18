"""
兴义文旅知识库 - 加载与检索模块
启动时加载 knowledge/ 目录下所有 JSON 数据到内存，
提供分类浏览和基于关键词的简单问答检索。
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger("xingyi-tour")

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"

# 分类目录 -> 中文展示名
CATEGORY_NAMES = {
    "attractions": "景点",
    "food": "美食",
    "hotels": "酒店住宿",
    "routes": "游玩路线",
    "transport": "交通",
    "activities": "活动体验",
    "culture": "民俗文化",
    "humanities": "人文历史",
    "reviews": "游客评价",
}

# 内存中的知识库: {category: [item, ...]}
STORE: dict[str, list[dict]] = {}


def load_knowledge() -> dict[str, int]:
    """
    加载 knowledge/ 目录下全部 JSON 数据文件到内存 STORE。

    输入：无
    输出：dict[str, int] - 各分类目录名到条目数量的映射，如 {"attractions": 11, "food": 10}
    作用：服务启动时一次性加载知识库，为后续检索提供内存数据；
          单个文件解析失败只记录日志，不影响整体加载。
    """
    STORE.clear()
    counts = {}
    if not KNOWLEDGE_DIR.exists():
        logger.warning(f"知识库目录不存在: {KNOWLEDGE_DIR}")
        return counts
    for category_dir in sorted(KNOWLEDGE_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        items = []
        for json_file in sorted(category_dir.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    item = json.load(f)
                item["_category"] = category
                items.append(item)
            except (json.JSONDecodeError, OSError) as e:
                logger.error(f"加载失败 {json_file.name}: {e}")
        STORE[category] = items
        counts[category] = len(items)
        logger.info(f"知识库分类 [{CATEGORY_NAMES.get(category, category)}] 加载 {len(items)} 条")
    total = sum(counts.values())
    logger.info(f"知识库加载完成，共 {total} 条数据")
    return counts


def list_categories() -> list[dict]:
    """
    返回知识库所有分类及其条目数。

    输入：无
    输出：list[dict] - 元素为 {"key": 分类目录名, "name": 中文展示名, "count": 条目数}
    作用：供前端/调用方浏览知识库分类结构。
    """
    return [
        {"key": cat, "name": CATEGORY_NAMES.get(cat, cat), "count": len(items)}
        for cat, items in STORE.items()
    ]


def list_items(category: str) -> list[dict]:
    """
    返回某分类下所有条目的简要信息。

    输入：category (str) - 分类目录名，如 "attractions"
    输出：list[dict] - 元素为 {"id", "name", "category"} 的简要条目列表
    作用：供分类浏览接口返回条目清单。
    """
    return [
        {"id": item.get("id"), "name": item.get("name"), "category": item.get("category")}
        for item in STORE.get(category, [])
    ]


def get_item(item_id: str) -> dict | None:
    """
    按 id 在全库中查找完整知识条目。

    输入：item_id (str) - 条目唯一标识，如 "attraction-wanfenglin"
    输出：dict | None - 找到则返回完整条目字典，否则返回 None
    作用：供条目详情查询及游客评价类条目的 target_id 关联查找。
    """
    for items in STORE.values():
        for item in items:
            if item.get("id") == item_id:
                return item
    return None


def search(question: str, limit: int = 3) -> list[dict]:
    """
    基于关键词匹配检索知识库（多维度权重打分）。

    输入：
        question (str) - 检索文本（通常为大模型 Function Calling 给出的关键词）
        limit (int) - 最多返回的条目数，默认 3
    输出：list[dict] - 按得分从高到低排序的命中条目列表
    作用：对每条知识的 name(+10) / aliases(+8) / tags(+5) / 分类(+3/+2) /
          description(+1) 打分，返回得分最高的条目，作为大模型的资料上下文。
    """
    question = question.strip()
    if not question:
        return []

    scored = []
    for items in STORE.values():
        for item in items:
            score = 0
            name = item.get("name", "")
            aliases = item.get("aliases", [])
            tags = item.get("tags", [])
            description = item.get("description", "")
            category = item.get("category", "")
            cat_display = CATEGORY_NAMES.get(item.get("_category", ""), "")

            # 名称完整命中权重最高
            if name and name in question:
                score += 10
            # 别名命中
            for alias in aliases:
                if alias and alias in question:
                    score += 8
            # 标签命中
            for tag in tags:
                if tag and tag in question:
                    score += 5
            # 分类词命中（如“美食”“酒店”“路线”）
            if cat_display and cat_display in question:
                score += 3
            if category and category in question:
                score += 2
            # 描述中的部分关键词命中
            for keyword in set(question.replace("？", " ").replace("?", " ").split()):
                if len(keyword) >= 2 and keyword in description:
                    score += 1

            if score > 0:
                scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:limit]]


def format_answer(item: dict) -> str:
    """
    把一条知识格式化为结构化的资料文本（供大模型阅读）。

    输入：item (dict) - 单条知识字典
    输出：str - 多行格式化文本，包含名称、描述、地址、开放时间、票价、提示等
    作用：作为 search_knowledge 工具的返回内容，让大模型依据准确资料作答；
          游客评价类条目结构不同，单独格式化。
    """
    # 游客评价类条目结构不同，单独格式化
    if item.get("_category") == "reviews":
        target = get_item(item.get("target_id", "")) or {}
        target_name = target.get("name", item.get("target_id", "相关条目"))
        parts = [f"【游客评价 · {target_name}】"]
        if item.get("rating"):
            parts.append(f"⭐ 评分：{item['rating']}")
        if item.get("content"):
            parts.append(f"💬 {item['content']}")
        if item.get("collected_platform"):
            parts.append(f"（来源：{item['collected_platform']}）")
        return "\n".join(parts)

    parts = [f"【{item.get('name', '未命名条目')}】"]
    if item.get("description"):
        parts.append(item["description"])

    location = item.get("location") or {}
    if location.get("address"):
        parts.append(f"📍 地址：{location['address']}")
    if location.get("transport_hint"):
        parts.append(f"🚌 交通提示：{location['transport_hint']}")

    time_info = item.get("time_info") or {}
    if time_info.get("opening_hours"):
        parts.append(f"🕐 开放时间：{time_info['opening_hours']}")

    price_info = item.get("price_info") or {}
    if price_info.get("ticket"):
        parts.append(f"🎫 门票：{price_info['ticket']}")
    if price_info.get("discount"):
        parts.append(f"💰 优惠政策：{price_info['discount']}")

    if item.get("suitable_for"):
        parts.append(f"👨‍👩‍👧 适合人群：{'、'.join(item['suitable_for'])}")

    notes = item.get("notes") or []
    if notes:
        parts.append("📌 提示：" + "；".join(notes))

    if item.get("data_status") == "待核实":
        parts.append("（注：部分信息待核实，出行前请以官方公告为准）")

    return "\n".join(parts)


def get_display_name(item: dict) -> str:
    """
    获取条目的人类可读展示名。

    输入：item (dict) - 单条知识字典
    输出：str - 展示名；游客评价类条目无 name 字段，
          则通过 target_id 关联目标条目，返回「游客评价 · 目标名」
    作用：供来源展示、日志记录等场景使用统一的名字。
    """
    if item.get("name"):
        return item["name"]
    target = get_item(item.get("target_id", "")) or {}
    return f"游客评价 · {target.get('name', item.get('target_id', ''))}"


def get_source_info(item: dict) -> dict:
    """
    提取知识条目的来源（source）与更新时间（update_time）信息。

    输入：item (dict) - 单条知识字典，原始 JSON 中须包含
          source_types（来源类型列表）/ collected_platform（评价采集平台）、
          collected_at（采集日期，格式 YYYY-MM-DD）
    输出：dict - {
            "id": 条目 id,
            "name": 展示名,
            "source": 人类可读来源描述（如「景区资料、网络公开资料」或「TripAdvisor（猫途鹰）」）,
            "update_time": 格式化后的核实日期（如「2026年10月16日」，无原始日期时为空字符串）
          }
    作用：为 /api/chat 响应的 sources 数组提供数据，前端据此渲染来源卡片。
    """
    # 来源：游客评价优先用采集平台名，其余用 source_types 拼接
    if item.get("collected_platform"):
        source = item["collected_platform"]
    else:
        source = "、".join(item.get("source_types", [])) or "兴义文旅知识库"

    # 更新时间：collected_at（YYYY-MM-DD）→ 「YYYY年M月D日」
    raw_date = item.get("collected_at", "") or ""
    update_time = ""
    try:
        year, month, day = raw_date.split("-")
        update_time = f"{int(year)}年{int(month)}月{int(day)}日"
    except (ValueError, AttributeError):
        update_time = raw_date

    return {
        "id": item.get("id", ""),
        "name": get_display_name(item),
        "source": source,
        "update_time": update_time,
    }
