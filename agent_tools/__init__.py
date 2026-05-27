"""
Agent 工具模块 — 所有可被 Agent 调用的工具和中间件。

导出内容：
  - web_search_tool    联网搜索（Tavily/DuckDuckGo）
  - code_exec_tool     代码执行沙箱
  - rag_summarize      RAG 知识库检索
  - create_mcp_client  MCP 客户端会话（async）
  - mcp_list_tools     列出 MCP Server 工具
  - mcp_call_tool      调用 MCP Server 工具
  - mcp_list_resources 列出 MCP Server 资源
  - mcp_read_resource  读取 MCP Server 资源

同步 MCP 工具（可直接注入 Agent）：
  - mcp_read_file      [文件系统] 读取文件内容
  - mcp_list_directory [文件系统] 列目录
  - mcp_search_files   [文件系统] 搜索文件
  - mcp_write_file     [文件系统] 写入文件
  - mcp_git_status     [Git] 查看仓库状态
  - mcp_git_log        [Git] 查看提交日志
  - mcp_git_diff       [Git] 查看文件差异
  - mcp_fetch_url      [Web] 获取网页内容

中间件单独由 agents/ 中各 Agent 按需导入。
"""

from .web_search_tool import web_search_tool
from .code_interpreter import code_exec_tool
from .rag_tool import rag_summarize
from .mcp_tool import create_mcp_client, mcp_list_tools, mcp_call_tool, mcp_list_resources, mcp_read_resource
from .mcp_tools import (
    mcp_read_file, mcp_list_directory, mcp_search_files, mcp_write_file,
    mcp_git_status, mcp_git_log, mcp_git_diff, mcp_fetch_url,
)

__all__ = [
    "web_search_tool",
    "code_exec_tool",
    "rag_summarize",
    "create_mcp_client",
    "mcp_list_tools",
    "mcp_call_tool",
    "mcp_list_resources",
    "mcp_read_resource",
    "mcp_read_file",
    "mcp_list_directory",
    "mcp_search_files",
    "mcp_write_file",
    "mcp_git_status",
    "mcp_git_log",
    "mcp_git_diff",
    "mcp_fetch_url",
]
