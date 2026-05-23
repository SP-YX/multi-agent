"""
联网搜索工具模块 — 为 Agent 提供实时互联网信息检索能力。

采用双引擎策略：
  1. Tavily（主）— 需要 API Key，返回结构化结果
  2. DuckDuckGo（备选）— 无需 Key，纯文本结果

具备自动重试机制（默认 2 次），引擎不可用时降级。
"""

import time
import logging
from typing import Optional
import warnings
from langchain_core.tools import tool

# 抑制 TavilySearchResults 弃用警告（将迁移到 langchain-tavily）
warnings.filterwarnings("ignore", message="The class `TavilySearchResults` was deprecated")

logger = logging.getLogger(__name__)

# 引擎可用性缓存，避免每次调用都尝试 import（None=未检测）
_TAVILY_AVAILABLE = None
_DUCKDUCKGO_AVAILABLE = None


def _check_tavily() -> bool:
    """
    检测 Tavily 搜索库是否可用（懒加载，结果缓存）。
    Returns: 可用返回 True
    """
    global _TAVILY_AVAILABLE
    if _TAVILY_AVAILABLE is not None:
        return _TAVILY_AVAILABLE
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        _TAVILY_AVAILABLE = True
    except ImportError:
        _TAVILY_AVAILABLE = False
    return _TAVILY_AVAILABLE


def _check_duckduckgo() -> bool:
    """
    检测 DuckDuckGo 搜索库是否可用（懒加载，结果缓存）。
    Returns: 可用返回 True
    """
    global _DUCKDUCKGO_AVAILABLE
    if _DUCKDUCKGO_AVAILABLE is not None:
        return _DUCKDUCKGO_AVAILABLE
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        _DUCKDUCKGO_AVAILABLE = True
    except ImportError:
        _DUCKDUCKGO_AVAILABLE = False
    return _DUCKDUCKGO_AVAILABLE


def _search_tavily(query: str, max_results: int = 3) -> str:
    """
    使用 Tavily API 执行搜索，返回结构化结果。
    Args:
        query: 搜索关键词
        max_results: 返回结果数量上限
    Returns: 格式化后的搜索结果文本
    """
    from langchain_community.tools.tavily_search import TavilySearchResults
    tool = TavilySearchResults(max_results=max_results)
    # _run() 返回 (results_list | error_str, raw_response) 元组
    result_tuple = tool.invoke({"query": query})
    results = result_tuple[0] if isinstance(result_tuple, tuple) else result_tuple
    if isinstance(results, str):
        return f"[Tavily Error] {results}"
    return _format_results(results, source="Tavily")


def _search_duckduckgo(query: str, max_results: int = 3) -> str:
    """
    使用 DuckDuckGo 执行搜索（无 API Key 要求）。
    Args:
        query: 搜索关键词
        max_results: 返回结果数量上限
    Returns: 格式化后的搜索结果文本
    """
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=max_results)
    tool = DuckDuckGoSearchRun(api_wrapper=wrapper)
    raw = tool.invoke(query)
    # DuckDuckGo 返回纯文本，包装成统一格式
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
    带重试的多引擎搜索：Tavily 优先 → DuckDuckGo 备选。
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
            if _check_tavily():
                return _search_tavily(query, max_results)
            elif _check_duckduckgo():
                return _search_duckduckgo(query, max_results)
            else:
                return "[Web Search] No search engine available. Install langchain-community with tavily or duckduckgo."
        except Exception as e:
            last_error = e
            logger.warning(f"Search attempt {attempt + 1} failed: {e}")
            if attempt < retries:
                time.sleep(1)  # 避让式等待
    return f"[Web Search] All attempts failed after {retries + 1} retries: {last_error}"


@tool(description="搜索互联网实时信息，返回结构化的搜索结果（标题、摘要、来源链接）")
def web_search(query: str, max_results: int = 3) -> str:
    """
    对外公开的搜索工具，LangChain @tool 装饰。
    支持 Tavily API 和 DuckDuckGo 双引擎 + 自动重试。
    Args:
        query: 搜索关键词
        max_results: 返回结果数量（1-10）
    Returns: 结构化搜索结果文本
    """
    return _search_with_retry(query, max_results)


# 导出为统一的工具变量名，方便 agent_tools/__init__.py 导入
web_search_tool = web_search
