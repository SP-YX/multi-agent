"""
MCP 同步工具封装 — 每步调用独立创建会话，避免事件循环冲突。
直接作为 LangChain @tool 注入 Agent 的 tools 列表即可。
"""

import asyncio
from langchain_core.tools import tool

from agent_tools.mcp_tool import create_mcp_client, mcp_call_tool


def _run_async(coro):
    """每次调用创建新事件循环，保证 MCP 会话生命周期完整"""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@tool(description="[MCP 文件系统] 读取文件内容。参数：path=文件路径")
def mcp_read_file(path: str) -> str:
    async def _():
        async with create_mcp_client("stdio", command="npx",
                                     args=["-y", "@modelcontextprotocol/server-filesystem", "."]) as c:
            return await mcp_call_tool(c, "read_file", {"path": path})
    return _run_async(_())


@tool(description="[MCP 文件系统] 读取目录列表。参数：path=目录路径")
def mcp_list_directory(path: str) -> str:
    async def _():
        async with create_mcp_client("stdio", command="npx",
                                     args=["-y", "@modelcontextprotocol/server-filesystem", "."]) as c:
            return await mcp_call_tool(c, "list_directory", {"path": path})
    return _run_async(_())


@tool(description="[MCP 文件系统] 搜索文件。参数：pattern=通配符模式")
def mcp_search_files(pattern: str) -> str:
    async def _():
        async with create_mcp_client("stdio", command="npx",
                                     args=["-y", "@modelcontextprotocol/server-filesystem", "."]) as c:
            return await mcp_call_tool(c, "search_files", {"pattern": pattern})
    return _run_async(_())


@tool(description="[MCP 文件系统] 写入文件。参数：path=路径, content=内容")
def mcp_write_file(path: str, content: str) -> str:
    async def _():
        async with create_mcp_client("stdio", command="npx",
                                     args=["-y", "@modelcontextprotocol/server-filesystem", "."]) as c:
            return await mcp_call_tool(c, "write_file", {"path": path, "content": content})
    return _run_async(_())


@tool(description="[MCP Git] 查看 Git 仓库状态")
def mcp_git_status() -> str:
    async def _():
        async with create_mcp_client("stdio", command="npx",
                                     args=["-y", "@modelcontextprotocol/server-git", "."]) as c:
            return await mcp_call_tool(c, "git_status", {})
    return _run_async(_())


@tool(description="[MCP Git] 查看 Git 提交日志。参数：max_commits=最大提交数（默认10）")
def mcp_git_log(max_commits: int = 10) -> str:
    async def _():
        async with create_mcp_client("stdio", command="npx",
                                     args=["-y", "@modelcontextprotocol/server-git", "."]) as c:
            return await mcp_call_tool(c, "git_log", {"max_commits": max_commits})
    return _run_async(_())


@tool(description="[MCP Git] 查看文件差异。参数：path=文件路径（可选）")
def mcp_git_diff(path: str = "") -> str:
    async def _():
        async with create_mcp_client("stdio", command="npx",
                                     args=["-y", "@modelcontextprotocol/server-git", "."]) as c:
            return await mcp_call_tool(c, "git_diff", {"path": path})
    return _run_async(_())


@tool(description="[MCP Web] 获取网页内容并转为 Markdown。参数：url=网页地址")
def mcp_fetch_url(url: str) -> str:
    async def _():
        async with create_mcp_client("stdio", command="npx",
                                     args=["-y", "@modelcontextprotocol/server-fetch"]) as c:
            return await mcp_call_tool(c, "fetch", {"url": url})
    return _run_async(_())


__all__ = [
    "mcp_read_file", "mcp_list_directory", "mcp_search_files", "mcp_write_file",
    "mcp_git_status", "mcp_git_log", "mcp_git_diff",
    "mcp_fetch_url",
]
