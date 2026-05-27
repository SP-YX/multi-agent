"""
Streamlit 界面 — 多智能体系统

功能：
  - 侧边栏会话列表（类似 DeepSeek/豆包，持久化保存）
  - 聊天式输入输出
  - 中间结果可展开查看（规划 / RAG / 搜索 / 代码）
  - 性能指标实时展示（工具调用次数、平均耗时）
  - Token 消耗预估

启动方式:
  streamlit run ui/app.py
"""

import os
import json
import uuid
import streamlit as st
from graph.agent_graph import agent_graph
from guardrails.guardrails import guardrails
from agent_tools.middleware import get_perf_stats, get_token_usage

SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "memory", "sessions")


# ═══════════════════════════════════════════════
# 会话持久化辅助函数
# ═══════════════════════════════════════════════

def _list_sessions() -> list[dict]:
    """扫描 memory/sessions/ 返回所有会话元数据（按最新消息排序）"""
    if not os.path.isdir(SESSIONS_DIR):
        return []
    sessions = []
    for f in os.listdir(SESSIONS_DIR):
        if not f.endswith(".json"):
            continue
        path = os.path.join(SESSIONS_DIR, f)
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        sid = data.get("session_id", f.replace(".json", ""))
        window = data.get("window", [])
        title = window[0]["user"][:30] if window else "(空会话)"
        latest_ts = window[-1]["timestamp"] if window else ""
        sessions.append({
            "session_id": sid,
            "title": title,
            "latest_ts": latest_ts,
            "message_count": len(window),
        })
    sessions.sort(key=lambda s: s["latest_ts"], reverse=True)
    return sessions


def _load_session_messages(session_id: str) -> list[dict]:
    """从 JSON 文件加载指定会话的聊天记录"""
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return []
    messages = []
    for turn in data.get("window", []):
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"], "intermediate": {}})
    return messages


def _delete_session_file(session_id: str):
    """删除指定会话的 JSON 文件"""
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(path):
        os.remove(path)


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
    st.session_state.messages = []
if "intermediate" not in st.session_state:
    st.session_state.intermediate = {}
if "show_delete_dialog" not in st.session_state:
    st.session_state.show_delete_dialog = None


def switch_session(session_id: str):
    """切换到指定会话：加载历史并更新 session_id"""
    st.session_state.session_id = session_id
    st.session_state.messages = _load_session_messages(session_id)
    st.session_state.intermediate = {}
    st.session_state.pending_delete = None


def new_session():
    """创建新会话：生成新 ID，清空聊天记录"""
    st.session_state.session_id = str(uuid.uuid4())[:8]
    st.session_state.messages = []
    st.session_state.intermediate = {}


def switch_session(session_id: str):
    """切换到指定会话：加载历史并更新 session_id"""
    st.session_state.session_id = session_id
    st.session_state.messages = _load_session_messages(session_id)
    st.session_state.intermediate = {}


@st.dialog("确认删除")
def confirm_delete_dialog(sid: str):
    """弹出确认删除对话框"""
    st.write(f"确定要删除此会话 **{sid[:8]}** 吗？此操作不可恢复。")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("确定", use_container_width=True, type="primary"):
            _delete_session_file(sid)
            st.session_state.show_delete_dialog = None
            if sid == st.session_state.session_id:
                new_session()
            st.rerun()
    with col2:
        if st.button("取消", use_container_width=True):
            st.session_state.show_delete_dialog = None
            st.rerun()


# ═══════════════════════════════════════════════
# 侧边栏
# ═══════════════════════════════════════════════

st.markdown("""
<style>
/* 删除按钮（✕）— 缩小到最小 */
div[data-testid="stSidebar"] div[data-testid="column"] button {
    font-size: 12px !important;
    min-height: 24px !important;
    height: 24px !important;
    padding: 0 2px !important;
    line-height: 24px !important;
    width: 28px !important;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("💬 会话记录")

    if st.button("＋ 新建会话", use_container_width=True, type="primary"):
        new_session()
        st.rerun()

    st.divider()

    sessions = _list_sessions()
    current_sid = st.session_state.session_id

    if not sessions:
        st.caption("暂无历史会话")
    else:
        for sess in sessions:
            sid = sess["session_id"]
            is_current = sid == current_sid
            label = sess["title"]
            if sess["message_count"] > 1:
                label += f" ({sess['message_count']}条)"

            cols = st.columns([5, 1])
            with cols[0]:
                btn = st.button(
                    label,
                    key=f"sess_{sid}",
                    use_container_width=True,
                    disabled=is_current,
                    type="secondary" if is_current else "tertiary",
                )
                if btn:
                    switch_session(sid)
                    st.rerun()
            with cols[1]:
                if st.button("✕", key=f"del_{sid}", help="删除此会话"):
                    st.session_state.show_delete_dialog = sid
                    st.rerun()

    st.divider()
    st.caption(f"当前: {current_sid}")

    st.divider()
    st.subheader("性能指标")

    perf = get_perf_stats()
    if perf:
        for name, stats in perf.items():
            st.metric(label=name, value=f"{stats['avg_ms']:.0f}ms", delta=f"{stats['count']}次调用")
    else:
        st.caption("暂无数据（运行任务后刷新）")

    token_usage = get_token_usage()
    st.metric("预估 Token 消耗", token_usage.get("total", 0))

    st.divider()
    st.subheader("💡 使用提示")
    st.markdown("""
    - 输入任务描述，系统自动调度
    - 五个 Agent 按流水线执行
    - 左侧可切换历史会话
    """)

# 弹出确认删除对话框（在侧边栏之外渲染）
if st.session_state.show_delete_dialog:
    confirm_delete_dialog(st.session_state.show_delete_dialog)


# ═══════════════════════════════════════════════
# 聊天历史渲染


# ═══════════════════════════════════════════════
# 聊天历史渲染
# ═══════════════════════════════════════════════

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
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
    import time as _time
    _t0 = _time.perf_counter()
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

            _elapsed = _time.perf_counter() - _t0
            placeholder.empty()
            status.update(label=f"处理完成（耗时 {_elapsed:.1f}s）", state="complete", expanded=False)
            st.markdown(final_cleaned)
            st.caption(f"⏱️ 耗时 {_elapsed:.1f}s")

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

    st.rerun()


# ═══════════════════════════════════════════════
# 页脚
# ═══════════════════════════════════════════════

st.divider()
st.caption(f"Session: {st.session_state.session_id} | 多智能体协作系统 v2.0")
