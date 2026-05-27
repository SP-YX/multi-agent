"""
MCP (Model Context Protocol) 工具模块 — 连接 MCP Server，动态发现并调用外部工具/资源。

支持两种传输模式：
  1. stdio — 通过子进程启动 MCP Server（本地）
  2. SSE  — 通过 HTTP 连接远程 MCP Server

用法示例:
  from agent_tools.mcp_tool import mcp_call_tool, mcp_list_tools, create_mcp_client

  async with create_mcp_client("stdio", command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "."]) as client:
      tools = await mcp_list_tools(client)
      result = await mcp_call_tool(client, "read_file", {"path": "/tmp/test.txt"})
"""

import os
import json
from typing import Any
from contextlib import asynccontextmanager

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client
    _MCP_AVAILABLE = True
except ImportError:
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None
    sse_client = None
    _MCP_AVAILABLE = False

MCP_TIMEOUT = int(os.getenv("MCP_TIMEOUT", "30"))


@asynccontextmanager
async def create_mcp_client(
    transport: str = "stdio",
    command: str | None = None,
    args: list[str] | None = None,
    url: str | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = MCP_TIMEOUT,
):
    """
    创建 MCP 客户端会话的异步上下文管理器。

    Args:
        transport: 传输协议，'stdio' 或 'sse'
        command: stdio 模式下 MCP Server 的可执行命令
        args: stdio 模式下的命令行参数
        url: SSE 模式下的 MCP Server URL
        headers: SSE 模式的自定义请求头
        timeout: 会话超时时间（秒）

    Yields:
        ClientSession: MCP 客户端会话实例

    Raises:
        RuntimeError: mcp 包未安装或参数错误
    """
    if not _MCP_AVAILABLE:
        raise RuntimeError("mcp 包未安装，请执行: pip install mcp")

    if transport == "stdio":
        if not command:
            raise ValueError("stdio 模式需要提供 command 参数")
        server_params = StdioServerParameters(command=command, args=args or [])
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
    elif transport == "sse":
        if not url:
            raise ValueError("SSE 模式需要提供 url 参数")
        async with sse_client(url=url, headers=headers, timeout=timeout) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
    else:
        raise ValueError(f"不支持的传输协议: {transport}")


async def mcp_list_tools(session: ClientSession) -> list[dict[str, Any]]:
    """
    列出 MCP Server 提供的所有工具。

    Args:
        session: MCP 客户端会话

    Returns:
        工具列表，每项包含 name、description、inputSchema
    """
    result = await session.list_tools()
    return [
        {
            "name": t.name,
            "description": t.description,
            "inputSchema": t.inputSchema,
        }
        for t in result.tools
    ]


async def mcp_call_tool(
    session: ClientSession,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> str:
    """
    调用 MCP Server 上的工具并返回结果文本。

    Args:
        session: MCP 客户端会话
        tool_name: 工具名称
        arguments: 工具参数

    Returns:
        工具执行结果的文本内容
    """
    result = await session.call_tool(tool_name, arguments=arguments or {})
    parts: list[str] = []
    for content in result.content:
        if content.type == "text":
            parts.append(content.text)
        elif content.type == "resource":
            parts.append(json.dumps(content.resource, ensure_ascii=False, indent=2))
        else:
            parts.append(str(content))
    return "\n".join(parts)


async def mcp_list_resources(session: ClientSession) -> list[dict[str, Any]]:
    """
    列出 MCP Server 提供的所有资源。

    Args:
        session: MCP 客户端会话

    Returns:
        资源列表，每项包含 uri、name、description、mimeType
    """
    result = await session.list_resources()
    return [
        {
            "uri": r.uri,
            "name": r.name,
            "description": r.description,
            "mimeType": r.mimeType,
        }
        for r in result.resources
    ]


async def mcp_read_resource(session: ClientSession, uri: str) -> str:
    """
    读取 MCP Server 上的指定资源。

    Args:
        session: MCP 客户端会话
        uri: 资源 URI

    Returns:
        资源内容文本
    """
    result = await session.read_resource(uri)
    parts: list[str] = []
    for content in result.contents:
        if hasattr(content, "text") and content.text:
            parts.append(content.text)
        elif hasattr(content, "blob") and content.blob:
            parts.append(f"[Binary resource {uri}]")
        else:
            parts.append(str(content))
    return "\n".join(parts)
