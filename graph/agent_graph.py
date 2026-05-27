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
  - run_agent() 统一异常兜底，保证单节点失败不阻塞流水线
  - _save_memory() 管理会话记忆的持久化
"""

import logging
import concurrent.futures
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from agents import RouterAgent, PlanAgent, RAGAgent, SearchAgent, CoderAgent, SummaryAgent
from agents.base_agent import BaseAgent
from memory.conversation_memory import ConversationMemory
from models.my_model import chat_model
from agent_tools.middleware import clear_memory_context

logger = logging.getLogger(__name__)

# 状态定义
class AgentState(TypedDict):
    """
    LangGraph 共享状态。
    每个 key 对应一个节点的输出,在pipeline中逐步填充
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


# Agent单例初始化
router_agent = RouterAgent()   # 路由Agent
plan_agent = PlanAgent()       # 计划Agent
rag_agent = RAGAgent()         # 检索召回Agent
search_agent = SearchAgent()   # 搜索Agent
coder_agent = CoderAgent()     # 编程Agent
summary_agent = SummaryAgent() # 总结Agent


# 记忆管理辅助函数
_memory_cache: dict[str, ConversationMemory] = {}

def _get_memory(state: AgentState) -> ConversationMemory:
    """
    根据 state 中的 session_id 获取对应会话记忆实例。
    同一个 session 内的多个图节点复用同一实例，避免反复读写 JSON。
    Args:
        state: 当前 Agent 状态
    Returns: ConversationMemory 实例
    """
    session_id = state.get("session_id", "default")
    if session_id not in _memory_cache:
        _memory_cache[session_id] = ConversationMemory(session_id=session_id)
    return _memory_cache[session_id]


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


def run_agent(agent: BaseAgent, state: AgentState, input_text: str, memory: ConversationMemory, **kwargs) -> str:
    """
    执行Agent。
    LLM 调用超时由模型层面的 request_timeout 参数处理，避免线程泄漏。
    
    Args:
        agent: Agent 实例
        state: 当前状态（暂未使用，保留一致性）
        input_text: 输入文本
        memory: 会话记忆
        **kwargs: 透传给 agent.run() 的额外参数
    Returns: Agent 输出或错误消息
    """
    context = memory.get_context()
    try:
        if context:
            agent.set_memory_context(context)
        result = agent.run(input_text, **kwargs)
        return result if result else "[Null]"
    except Exception as e:
        logger.error(f"[run_agent] Agent执行失败! 错误:{e}", exc_info=True)
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
    import time as _time
    _t0 = _time.perf_counter()
    logger.info(f"[router_node] start | query={state['query'][:50]}")
    memory = _get_memory(state)
    _t1 = _time.perf_counter()
    result = run_agent(router_agent, state, state["query"], memory)
    _t2 = _time.perf_counter()
    valid = {"simple", "code", "rag", "search", "complex"}
    if result not in valid:
        result = "complex"
    logger.info(f"[router_node] done | get_memory={_t1-_t0:.2f}s run_agent={_t2-_t1:.2f}s route={result}")
    return {"route": result}

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
    context = memory.get_context()
    query = state["query"]
    messages = [("system", "你是一个友好的 AI 助手，请用简洁直接的方式回答用户的问题。回答控制在 5 句话以内。")]
    if context:
        messages.append(("human", f"历史对话：{context}"))
    messages.append(("human", query))
    resp = chat_model.invoke(messages)
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
    result = run_agent(plan_agent, state, state["query"], memory)
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
        return run_agent(rag_agent, state, query, mem)

    def run_search():
        mem = _get_memory(state)
        return run_agent(search_agent, state, query, mem)

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
    result = run_agent(coder_agent, state, query, memory)
    return {"code_result": result}


def summary_node(state: AgentState) -> dict:
    """
    【节点 5】结果汇总节点。
    汇聚 plan + RAG + search + code 的全部输出，
    生成统一、完整的最终回答。同时将会话保存到记忆系统。
    """
    logger.info("[summary_node] start")
    memory = _get_memory(state)
    result = run_agent(
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
