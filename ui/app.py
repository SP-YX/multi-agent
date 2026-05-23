"""
Streamlit Web 界面 — 多智能体系统的可视化操作面板。

功能：
  - 聊天式输入输出
  - 中间结果可展开查看（规划 / RAG / 搜索 / 代码）
  - 会话管理（自动生成 session_id，支持重置）
  - 性能指标实时展示（工具调用次数、平均耗时）
  - Token 消耗预估

启动方式：
  streamlit run ui/app.py
"""

import uuid
import concurrent.futures
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agents import PlanAgent, RAGAgent, SearchAgent, CoderAgent, SummaryAgent
from memory.conversation_memory import ConversationMemory
from graph.agent_graph import _safe_run
from agent_tools.middleware import get_perf_stats, get_token_usage

# Agent 单例
plan_agent = PlanAgent()
rag_agent = RAGAgent()
search_agent = SearchAgent()
coder_agent = CoderAgent()
summary_agent = SummaryAgent()


# ═══════════════════════════════════════════════
# 页面配置
# ═══════════════════════════════════════════════

st.set_page_config(
    page_title="Multi-Agent System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🤖 多智能体协作系统")
st.markdown("基于 **LangGraph** 的多智能体编排引擎 — 规划 → RAG → 搜索 → 编码 → 汇总")


# ═══════════════════════════════════════════════
# Session State 初始化
# ═══════════════════════════════════════════════

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []       # 聊天历史（用于界面渲染）
if "intermediate" not in st.session_state:
    st.session_state.intermediate = {}   # 上次执行中间结果缓存


def reset_session():
    """重置会话：生成新 ID，清空聊天记录和中间结果。"""
    st.session_state.session_id = str(uuid.uuid4())[:8]
    st.session_state.messages = []
    st.session_state.intermediate = {}


# ═══════════════════════════════════════════════
# 侧边栏
# ═══════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ 控制面板")

    st.text_input("Session ID", value=st.session_state.session_id, disabled=True)
    if st.button("🔄 新建会话", use_container_width=True):
        reset_session()
        st.rerun()

    st.divider()
    st.subheader("📊 性能指标")

    # 从全局中间件读取性能数据
    perf = get_perf_stats()
    if perf:
        for name, stats in perf.items():
            st.metric(
                label=name,
                value=f"{stats['avg_ms']:.0f}ms",
                delta=f"{stats['count']}次调用",
            )
    else:
        st.caption("暂无数据（运行任务后刷新）")

    token_usage = get_token_usage()
    st.metric("预估 Token 消耗", token_usage.get("total", 0))

    st.divider()
    st.subheader("💡 使用提示")
    st.markdown("""
    - 输入任务描述，系统自动调度
    - 五个 Agent 按流水线执行
    - 点击展开可查看中间结果
    """)


# ═══════════════════════════════════════════════
# 聊天历史渲染
# ═══════════════════════════════════════════════

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # 如果是助手回复且有中间结果，显示可展开区域
        if msg["role"] == "assistant" and msg.get("intermediate"):
            with st.expander("🔍 查看中间结果"):
                for key, value in msg["intermediate"].items():
                    if value:
                        st.text(f"[{key}]")
                        st.text(value[:500] + ("..." if len(value) > 500 else ""))


# ═══════════════════════════════════════════════
# 用户输入与任务执行
# ═══════════════════════════════════════════════

if prompt := st.chat_input("输入你的任务..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("🤔 正在处理...", expanded=True)
        placeholder = st.empty()

        try:
            memory = ConversationMemory(session_id=st.session_state.session_id)
            query = prompt
            state = {"query": query, "session_id": st.session_state.session_id}

            # 1. 规划
            status.update(label="📋 规划任务...", state="running")
            sub_tasks = _safe_run(plan_agent, state, query, memory)
            placeholder.text_area("📋 规划结果", sub_tasks[:200], height=80)

            # 2. RAG + Search 并行
            status.update(label="📚 检索知识库 + 🌐 搜索网络（并行）...", state="running")
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                fut_rag = pool.submit(_safe_run, rag_agent, state, sub_tasks or query, memory)
                fut_search = pool.submit(_safe_run, search_agent, state, sub_tasks or query, memory)
                concurrent.futures.wait([fut_rag, fut_search])
            rag_result = fut_rag.result()
            search_result = fut_search.result()
            placeholder.empty()

            # 3. 代码
            status.update(label="💻 执行代码...", state="running")
            code_result = _safe_run(coder_agent, state, query, memory)

            # 4. 汇总
            status.update(label="📝 汇总结果...", state="running")
            final_answer = _safe_run(
                summary_agent, state, query, memory,
                sub_tasks=sub_tasks, rag_res=rag_result,
                search_res=search_result, code_res=code_result,
            )

            status.update(label="✅ 处理完成", state="complete", expanded=False)

            st.markdown(final_answer)

            intermediate = {
                "sub_tasks": sub_tasks, "rag_result": rag_result,
                "search_result": search_result, "code_result": code_result,
            }
            with st.expander("🔍 查看中间结果"):
                cols = st.columns(2)
                with cols[0]:
                    st.text_area("📋 任务规划", sub_tasks, height=150)
                    st.text_area("🌐 搜索结果", search_result, height=150)
                with cols[1]:
                    st.text_area("📚 RAG 结果", rag_result, height=150)
                    st.text_area("💻 代码执行", code_result, height=150)

            st.session_state.messages.append({
                "role": "assistant", "content": final_answer,
                "intermediate": intermediate,
            })
            st.session_state.intermediate = intermediate

        except Exception as e:
            status.update(label="❌ 处理失败", state="error")
            st.error(f"处理失败: {type(e).__name__}: {e}")


# ═══════════════════════════════════════════════
# 页脚
# ═══════════════════════════════════════════════

st.divider()
st.caption(f"Session: {st.session_state.session_id} | 多智能体协作系统 v2.0")
