"""
联网搜索工具模块 — 为 Agent 提供实时互联网信息检索能力。

采用双引擎策略：
  1. Tavily(主)— 需要 API Key,返回结构化结果
  2. DuckDuckGo(备选)— 无需 Key,纯文本结果

具备自动重试机制（默认 2 次），引擎不可用时降级。
"""

import os
import time
from utils.log_tool import logger
import warnings
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv(override=True) 

# 抑制 TavilySearchResults 弃用警告（将迁移到 langchain-tavily）
warnings.filterwarnings("ignore", message="The class `TavilySearchResults` was deprecated")

_tavily_available = True # TavilySearchResults是否可用(优先使用)
try:
    from langchain_community.tools.tavily_search import TavilySearchResults
except ImportError:
    TavilySearchResults = None 
    _tavily_available = False

_duckduckgo_available = True # DuckDuckGoSearch相关包是否可用
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
except ImportError:
    DuckDuckGoSearchRun = None
    DuckDuckGoSearchAPIWrapper = None
    _duckduckgo_available = False


def _search_tavily(query: str, max_results: int = 3) -> str:
    wrapper = TavilySearchResults(max_results=max_results, api_key = os.getenv("TAVILY_API_KEY"))
    result_tuple = wrapper.invoke({"query": query})
    results = result_tuple[0] if isinstance(result_tuple, tuple) else result_tuple
    if isinstance(results, str):
        return f"[Tavily Error] {results}"
    return _format_results(results, source="Tavily")


def _search_duckduckgo(query: str, max_results: int = 3) -> str:
    api_wrapper = DuckDuckGoSearchAPIWrapper(max_results=max_results)
    tool = DuckDuckGoSearchRun(api_wrapper=api_wrapper)
    raw = tool.invoke(query)
    return _format_results([{"title": "Result", "content": raw, "link": ""}], source="DuckDuckGo")


def _format_results(results: list, source: str) -> str:
    """
    将搜索结果统一格式化为 Markdown 文本，便于 LLM 解析。
    Args:
        results: 搜索结果列表（每条含 title/content/link）
        source: 搜索引擎名称
    Returns: 格式化的 Markdown 文本
    """
    if not results:
        return f"[{source}] No results found."
    lines = [f"## Web Search Results ({source})"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        content = r.get("content", r.get("snippet", ""))
        link = r.get("link", "")
        lines.append(f"\n### [{i}] {title}")
        lines.append(f"   {content[:500]}")       # 截断过长内容
        if link:
            lines.append(f"   Source: {link}")
    return "\n".join(lines)


def _search_with_retry(query: str, max_results: int = 3, retries: int = 2) -> str:
    """
    多引擎搜索: Tavily 优先 → DuckDuckGo 备选。
    每次失败后等待 1s 再重试，全部失败返回错误消息。
    Args:
        query: 搜索关键词
        max_results: 结果数量上限
        retries: 单个引擎重试次数
    Returns: 搜索结果或错误消息
    """
    last_error = None
    for attempt in range(retries + 1):
        try:
            if _tavily_available: # 优先Tavily
                return _search_tavily(query, max_results)
            if _duckduckgo_available:
                return _search_duckduckgo(query, max_results)
            
            info = f"[_search_with_retry] 联网搜索失败! 缺少包Tavily或Duckduckgo"
            logger.warning(info)
            return info
        except Exception as e:
            last_error = e
            logger.warning(f"[_search_with_retry] 联网搜索{attempt + 1}次失败: {e}")
            if attempt < retries:
                time.sleep(1)

    error = f"[_search_with_retry] 联网搜索失败! 最新失败消息:{last_error}"
    return error


@tool(description="搜索互联网实时信息，返回结构化的搜索结果（标题、摘要、来源链接）")
def web_search_tool(query: str, max_results: int = 3) -> str:
    """
    对外公开的搜索工具
    优先使用Tavily API, DuckDuckGo作为备选, 自动重试
    Args:
        query: 搜索关键词
        max_results: 返回结果数量
    Returns: 结构化搜索结果文本
    """
    return _search_with_retry(query, max_results)
