"""
Agent 中间件模块 — 基于 LangChain Middleware 机制实现横切关注点分离。

中间件类型：
  1. @wrap_tool_call — 在工具调用前后执行（类似装饰器，可嵌套）
  2. @before_model   — 在调用 LLM 之前执行

当前注册的中间件（共 6 个）：

  Tool 中间件（执行顺序 = 注册顺序）：
    monitor_tool              → 工具调用日志记录
    performance_middleware    → 工具耗时统计
    error_handler_middleware  → 工具异常友好兜底

  Pre-Model 中间件：
    log_before_model          → 模型调用前日志
    memory_inject_middleware  → 注入会话记忆上下文
    token_counter_middleware  → Token 消耗预估

同时通过模块级全局变量对外暴露性能与 Token 统计数据。
"""

import time
import json
from typing import Optional, Callable

from langchain.agents.middleware import AgentState, Runtime, before_model, wrap_tool_call
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from utils.log_tool import logger


# ═══════════════════════════════════════════════
# 全局状态区域（跨 Agent 共享的统计数据）
# ═══════════════════════════════════════════════

# 工具性能统计：{工具名: [耗时1_ms, 耗时2_ms, ...]}
_PERF_STATS: dict[str, list[float]] = {}

# Token 计数器
_TOKEN_COUNTER: dict[str, int] = {"prompt": 0, "completion": 0, "total": 0}

# 当前会话的记忆上下文（由 graph/agent_graph.py 在节点执行前 set）
_MEMORY_CONTEXT: Optional[str] = None


def set_memory_context(context: str):
    """
    设置当前会话的记忆上下文（由编排层在 Agent 执行前调用）。
    Args:
        context: 格式化的记忆文本
    """
    global _MEMORY_CONTEXT
    _MEMORY_CONTEXT = context


def clear_memory_context():
    """清除当前会话的记忆上下文（Agent 执行完毕后调用）。"""
    global _MEMORY_CONTEXT
    _MEMORY_CONTEXT = None


def get_perf_stats() -> dict:
    """
    获取工具性能统计摘要。
    Returns: {工具名: {count, total_ms, avg_ms, max_ms}}
    """
    result = {}
    for name, durations in _PERF_STATS.items():
        result[name] = {
            "count": len(durations),
            "total_ms": round(sum(durations), 2),
            "avg_ms": round(sum(durations) / len(durations), 2) if durations else 0,
            "max_ms": round(max(durations), 2) if durations else 0,
        }
    return result


def get_token_usage() -> dict:
    """
    获取 Token 消耗预估。
    Returns: {prompt: int, completion: int, total: int}
    """
    return dict(_TOKEN_COUNTER)


# ═══════════════════════════════════════════════
# Tool-call 中间件（在工具被调用时执行）
# ═══════════════════════════════════════════════

@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """
    工具调用监控中间件：记录工具名称、输入参数、调用结果。
    异常时打印错误日志并向上传播异常。
    """
    logger.debug(f"[monitor_tool] 执行工具: {request.tool_call['name']}")
    logger.debug(f"[monitor_tool] 传入参数: {request.tool_call['args']}")
    try:
        res = handler(request)
        logger.debug(f"[monitor_tool] 工具 {request.tool_call['name']} 调用成功!")
        return res
    except Exception as e:
        logger.error(f"[monitor_tool] 工具 {request.tool_call['name']} 调用失败: {e}")
        raise


@wrap_tool_call
def performance_middleware(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """
    性能监控中间件：记录每个工具的调用耗时，汇总到全局 _PERF_STATS。
    使用 time.perf_counter() 获得高精度计时。
    """
    tool_name = request.tool_call["name"]
    start = time.perf_counter()
    try:
        res = handler(request)
        return res
    finally:
        # finally 确保无论是否异常都记录耗时
        elapsed = (time.perf_counter() - start) * 1000  # 秒→毫秒
        _PERF_STATS.setdefault(tool_name, []).append(elapsed)
        logger.debug(f"[performance] {tool_name} 耗时: {elapsed:.1f}ms")


@wrap_tool_call
def error_handler_middleware(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """
    错误处理中间件：工具异常时捕获并返回友好的 ToolMessage，
    避免异常向上传播导致 Agent 崩溃。同时通知 LLM 尝试替代方案。
    """
    try:
        return handler(request)
    except Exception as e:
        logger.error(f"[error_handler] 工具 {request.tool_call['name']} 异常: {e}")
        return ToolMessage(
            content=f"[System] Tool '{request.tool_call['name']}' encountered an error: {e}. "
                    f"Please inform the user and try an alternative approach.",
            tool_call_id=request.tool_call.get("id", ""),
        )


# ═══════════════════════════════════════════════
# Pre-Model 中间件（在 LLM 被调用前执行）
# ═══════════════════════════════════════════════

@before_model
def log_before_model(state: AgentState, runtime: Runtime):
    """
    模型调用前日志中间件：记录消息数量和最后一条消息内容。
    帮助调试 Agent 的消息历史是否正常构建。
    """
    msg_count = len(state["messages"])
    logger.debug(f"[log_before_model] 即将调用模型: {msg_count} 条消息")
    if msg_count > 0:
        last = state["messages"][-1]
        logger.debug(f"[log_before_model] {type(last).__name__}: {last.content[:200]}")
    return None


@before_model
def memory_inject_middleware(state: AgentState, runtime: Runtime):
    """
    记忆注入中间件：将全局 _MEMORY_CONTEXT 作为 SystemMessage 插入到
    state["messages"] 中，使 LLM 能感知到历史对话内容。
    上下文由 graph/agent_graph.py / ui/app.py 在 Agent 执行前通过
    set_memory_context() 设置。
    """
    if _MEMORY_CONTEXT:
        from langchain_core.messages import SystemMessage
        state["messages"].insert(0, SystemMessage(content=_MEMORY_CONTEXT))
        logger.debug(f"[memory_inject] 注入记忆上下文 ({len(_MEMORY_CONTEXT)} chars)")
    return None


@before_model
def token_counter_middleware(state: AgentState, runtime: Runtime):
    """
    Token 计数中间件：基于字符数粗略估算 LLM 调用的 Token 消耗。
    估算公式：总字符数 ÷ 2（中英文混合场景的经验值）。
    注意：这只是预估算，不是精确计数。
    """
    prompt_chars = 0
    for msg in state["messages"]:
        prompt_chars += len(str(msg.content))
    estimated_tokens = prompt_chars // 2
    _TOKEN_COUNTER["prompt"] += estimated_tokens
    _TOKEN_COUNTER["total"] += estimated_tokens
    logger.debug(f"[token_counter] 本轮预估 tokens: {estimated_tokens}")
    return None
