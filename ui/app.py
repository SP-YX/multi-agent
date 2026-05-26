"""
Streamlit 界面 — 多智能体系统

功能：
  - 聊天式输入输出
  - 中间结果可展开查看（规划 / RAG / 搜索 / 代码）
  - 会话管理（自动生成 session_id,支持重置）
  - 性能指标实时展示（工具调用次数、平均耗时）
  - Token 消耗预估

启动方式(命令):
  streamlit run ui/app.py
"""

import uuid
import streamlit as st
from graph.agent_graph import agent_graph
from guardrails.guardrails import guardrails
from agent_tools.middleware import get_perf_stats, get_token_usage


 

# ═══════════════════════════════════════════════
# 页面配置
# ═══════════════════════════════════════════════

st.set_page_config(
    page_title="Multi-Agent System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("多智能体协作系统")
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
    st.header("控制面板")

    st.text_input("Session ID", value=st.session_state.session_id, disabled=True)
    if st.button("新建会话", use_container_width=True):
        reset_session()
        st.rerun()

    st.divider()
    st.subheader("性能指标")

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
            with st.expander("查看中间结果"):
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
        status = st.status("正在处理...", expanded=True)
        placeholder = st.empty()

        try:
            ok, cleaned, violations = guardrails.pre_process(prompt, st.session_state.session_id)
            if not ok:
                msg = violations[0].get("message", "输入被拦截")
                st.error(f"[Guardrails Blocked] {msg}")
                st.session_state.messages.append({"role": "assistant", "content": f"[Guardrails Blocked] {msg}"})
                st.stop()

            query = cleaned
            initial_state = {"query": query, "session_id": st.session_state.session_id}
            final_answer = ""
            intermediate = {}

            for event in agent_graph.stream(initial_state, stream_mode="updates"):
                for node_name, update in event.items():
                    if node_name == "router":
                        status.update(label="分析问题类型...", state="running")
                        st.caption(f"路由结果: {update.get('route', 'complex')}")
                    elif node_name == "simple_reply":
                        status.update(label="快速回复...", state="running")
                    elif node_name == "planner":
                        status.update(label="规划任务...", state="running")
                        if update.get("sub_tasks"):
                            placeholder.text_area("规划结果", update["sub_tasks"][:200], height=80)
                    elif node_name == "retrieval":
                        status.update(label="检索知识库 + 搜索网络（并行）...", state="running")
                        placeholder.empty()
                    elif node_name == "coder":
                        status.update(label="执行代码...", state="running")
                    elif node_name == "summary":
                        status.update(label="汇总结果...", state="running")

                    intermediate.update(update)
                    if "final_answer" in update:
                        final_answer = update["final_answer"]

            ok_out, final_cleaned, out_violations = guardrails.post_process(final_answer)
            if not ok_out:
                st.error(f"[Guardrails Blocked] {out_violations[0].get('message', '输出被拦截')}")
                st.session_state.messages.append({"role": "assistant", "content": "[Guardrails Blocked] 输出被拦截"})
                st.stop()

            placeholder.empty()
            status.update(label="处理完成", state="complete", expanded=False)
            st.markdown(final_cleaned)

            has_intermediate = any(k in intermediate for k in ("sub_tasks", "rag_result", "search_result", "code_result"))
            if has_intermediate:
                with st.expander("查看中间结果"):
                    cols = st.columns(2)
                    with cols[0]:
                        if intermediate.get("sub_tasks"):
                            st.text_area("任务规划", intermediate["sub_tasks"], height=150)
                        if intermediate.get("search_result"):
                            st.text_area("搜索结果", intermediate["search_result"], height=150)
                    with cols[1]:
                        if intermediate.get("rag_result"):
                            st.text_area("RAG 结果", intermediate["rag_result"], height=150)
                        if intermediate.get("code_result"):
                            st.text_area("代码执行", intermediate["code_result"], height=150)

            st.session_state.messages.append({
                "role": "assistant", "content": final_cleaned,
                "intermediate": intermediate if has_intermediate else {},
            })
            st.session_state.intermediate = intermediate if has_intermediate else {}

        except Exception as e:
            status.update(label="❌ 处理失败", state="error")
            st.error(f"处理失败: {type(e).__name__}: {e}")


# ═══════════════════════════════════════════════
# 页脚
# ═══════════════════════════════════════════════

st.divider()
st.caption(f"Session: {st.session_state.session_id} | 多智能体协作系统 v2.0")
