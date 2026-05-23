"""
LangGraph 状态图编排 — 多智能体协作系统的核心调度引擎。

流程拓扑：
  router → 根据问题类型分流：
    simple  → 直接回复（一次 LLM 调用）
    code    → coder → summary
    rag     → retrieval → summary
    search  → retrieval → summary
    complex → planner → retrieval → coder → summary

RAG 和 Search 互相独立，通过 ThreadPoolExecutor 并行执行以节省时间。

关键设计：
  - router 前置分类，简单问题不走全流水线
  - _safe_run() 统一异常兜底，保证单节点失败不阻塞流水线
  - _inject_memory() / _save_memory() 管理会话记忆的注入与持久化
"""

import time
import logging
import concurrent.futures
from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END
from agents import PlanAgent, RAGAgent, SearchAgent, CoderAgent, SummaryAgent
from memory.conversation_memory import ConversationMemory
from models.my_model import chat_model
from agent_tools.middleware import (
    set_memory_context,
    clear_memory_context,
    get_perf_stats,
    get_token_usage,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
# 状态定义
# ═══════════════════════════════════════════════

class AgentState(TypedDict):
    """
    LangGraph 的共享状态结构。
    每个 key 对应一个节点的输出，在流水线中逐步填充。
    """
    query: str              # 用户原始输入
    route: str              # 路由器分类结果
    sub_tasks: str          # PlanAgent → 任务拆解计划
    rag_result: str         # RAGAgent → 知识库检索结果
    search_result: str      # SearchAgent → 联网搜索结果
    code_result: str        # CoderAgent → 代码执行结果
    final_answer: str       # SummaryAgent → 最终汇总回答
    error: Optional[str]    # 节点执行时的异常信息（暂未使用，保留扩展）
    session_id: str         # 会话标识，用于记忆隔离


# ═══════════════════════════════════════════════
# Agent 单例初始化（全局共享，避免重复创建）
# ═══════════════════════════════════════════════

plan_agent = PlanAgent()
rag_agent = RAGAgent()
search_agent = SearchAgent()
coder_agent = CoderAgent()
summary_agent = SummaryAgent()


# ═══════════════════════════════════════════════
# 记忆管理辅助函数
# ═══════════════════════════════════════════════

def _get_memory(state: AgentState) -> ConversationMemory:
    """
    根据 state 中的 session_id 获取对应会话记忆实例。
    Args:
        state: 当前 Agent 状态
    Returns: ConversationMemory 实例
    """
    session_id = state.get("session_id", "default")
    return ConversationMemory(session_id=session_id)


def _inject_memory(memory: ConversationMemory):
    """
    将记忆上下文注入全局中间件变量。
    由 memory_inject_middleware 在 pre-model 阶段读取并添加到 AgentState。
    Args:
        memory: 当前会话的记忆实例
    """
    context = memory.get_context()
    if context:
        set_memory_context(context)


def _save_memory(memory: ConversationMemory, query: str, answer: str):
    """
    将本轮对话保存到记忆系统。
    通常在 summary_node 执行完毕后调用。
    Args:
        memory: 当前会话的记忆实例
        query: 用户提问
        answer: 最终回答
    """
    memory.add_conversation(user=query, assistant=answer)


def _safe_run(agent, state: AgentState, input_text: str, memory: ConversationMemory, **kwargs) -> str:
    """
    统一的安全执行包装器。
    1. 注入记忆上下文
    2. 执行 Agent
    3. 捕获异常返回错误消息
    4. 清理记忆上下文
    Args:
        agent: Agent 实例
        state: 当前状态（未使用，保留一致性）
        input_text: 输入文本
        memory: 会话记忆
        **kwargs: 透传给 agent.run() 的额外参数
    Returns: Agent 输出或错误消息
    """
    _inject_memory(memory)
    try:
        result = agent.run(input_text, **kwargs)
        return result if result else "[No output]"
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        return f"[System Error] {type(e).__name__}: {e}"
    finally:
        clear_memory_context()


# ═══════════════════════════════════════════════
# 路由器 — 根据问题类型选择执行路径
# ═══════════════════════════════════════════════

def router_node(state: AgentState) -> dict:
    """
    【节点 0】路由器：将用户问题分类，决定走哪条流水线。
    用一次轻量 LLM 调用完成分类，避免所有问题都走全量流程。
    """
    query = state["query"]
    prompt = (
        f"将以下问题分类到其中一个类别（只返回类别名称）：\n"
        f"- simple：问候、闲聊、简单问答（一句话能回答的简单问题）\n"
        f"- code：编程、算法、代码编写与调试\n"
        f"- rag：需要查找本地知识库或内部文档资料\n"
        f"- search：实时信息、新闻、最新动态或网络内容\n"
        f"- complex：复杂任务，需要先规划再执行的多步骤工作\n\n"
        f"问题：{query}\n"
        f"类别："
    )
    resp = chat_model.invoke(prompt)
    route = resp.content.strip().lower()
    valid = {"simple", "code", "rag", "search", "complex"}
    if route not in valid:
        route = "complex"
    logger.info(f"[router] {query[:50]} -> {route}")
    return {"route": route}


def route_decision(state: AgentState) -> str:
    """路由器条件边：根据 route 字段跳转到对应节点。"""
    return state.get("route", "complex")


def after_retrieval(state: AgentState) -> str:
    """检索节点后的条件边：rag/search 直接汇总，complex 继续执行代码。"""
    route = state.get("route", "complex")
    return "summary" if route in ("rag", "search") else "coder"


# ═══════════════════════════════════════════════
# 状态图节点函数
# ═══════════════════════════════════════════════

def simple_reply_node(state: AgentState) -> dict:
    """
    【节点 1】简单回复节点。
    不经过任何 Agent，直接调用 LLM 生成回答并保存记忆。
    """
    logger.info("[simple_reply_node] 直接回复")
    memory = _get_memory(state)
    resp = chat_model.invoke(state["query"])
    answer = resp.content
    _save_memory(memory, state["query"], answer)
    return {"final_answer": answer}


def planner_node(state: AgentState) -> dict:
    """
    【节点 1】任务规划节点。
    将用户原始需求拆解为结构化执行计划。
    """
    logger.info(f"[planner_node] start | query={state['query'][:50]}")
    memory = _get_memory(state)
    result = _safe_run(plan_agent, state, state["query"], memory)
    return {"sub_tasks": result}


def retrieval_node(state: AgentState) -> dict:
    """
    【节点 2】检索节点：并行执行 RAG（本地知识库）+ Search（联网搜索）。
    二者互相独立，同时执行以节省约一半时间。
    """
    logger.info("[retrieval_node] 并行启动 RAG + Search")
    query = state.get("sub_tasks", "") or state["query"]

    def run_rag():
        mem = _get_memory(state)
        return _safe_run(rag_agent, state, query, mem)

    def run_search():
        mem = _get_memory(state)
        return _safe_run(search_agent, state, query, mem)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fut_rag = pool.submit(run_rag)
        fut_search = pool.submit(run_search)
        concurrent.futures.wait([fut_rag, fut_search])

    return {
        "rag_result": fut_rag.result(),
        "search_result": fut_search.result(),
    }


def code_node(state: AgentState) -> dict:
    """
    【节点 4】代码执行节点。
    如需要，生成并执行 Python 代码（计算/处理/转换）。
    """
    logger.info("[code_node] start")
    memory = _get_memory(state)
    query = state["query"]
    result = _safe_run(coder_agent, state, query, memory)
    return {"code_result": result}


def summary_node(state: AgentState) -> dict:
    """
    【节点 5】结果汇总节点。
    汇聚 plan + RAG + search + code 的全部输出，
    生成统一、完整的最终回答。同时将会话保存到记忆系统。
    """
    logger.info("[summary_node] start")
    memory = _get_memory(state)
    result = _safe_run(
        summary_agent, state, state["query"],
        memory,
        sub_tasks=state.get("sub_tasks", ""),
        rag_res=state.get("rag_result", ""),
        search_res=state.get("search_result", ""),
        code_res=state.get("code_result", ""),
    )
    # 汇总完成后保存对话到记忆
    _save_memory(memory, state["query"], result)
    return {"final_answer": result}


# ═══════════════════════════════════════════════
# 工作流构建
# ═══════════════════════════════════════════════

def build_agent_graph():
    """
    构建 LangGraph 状态图。
    流程：router → 按类型分流
      simple  → 直接回复
      code    → coder → summary
      rag     → retrieval → summary
      search  → retrieval → summary
      complex → planner → retrieval → coder → summary
    Returns: 编译后的可执行图
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("simple_reply", simple_reply_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("coder", code_node)
    workflow.add_node("summary", summary_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "simple": "simple_reply",
            "code": "coder",
            "rag": "retrieval",
            "search": "retrieval",
            "complex": "planner",
        },
    )

    workflow.add_edge("simple_reply", END)
    workflow.add_edge("planner", "retrieval")

    workflow.add_conditional_edges(
        "retrieval",
        after_retrieval,
        {
            "coder": "coder",
            "summary": "summary",
        },
    )

    workflow.add_edge("coder", "summary")
    workflow.add_edge("summary", END)

    return workflow.compile()


# 全局单例图实例
agent_graph = build_agent_graph()
