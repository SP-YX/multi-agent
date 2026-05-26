"""
CLI 入口 — 多智能体系统的命令行交互界面。

通过标准输入读取用户任务，逐轮调用 LangGraph 流水线，
输出最终回答，并在退出时打印性能统计。
"""

import os
import uuid
import logging
from dotenv import load_dotenv

# 抑制 Streamlit 在 CLI 模式下产生的 "missing ScriptRunContext" 警告
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)

from guardrails import Guardrails
from agent_tools.middleware import get_perf_stats, get_token_usage

# 加载 .env 中的 API Key 等环境变量
load_dotenv()


def main():
    """主循环：读取用户输入 → Guardrails 校验 → 调用 Agent Graph → 打印结果。"""
    session_id = str(uuid.uuid4())[:8]
    guard = Guardrails()

    print("=" * 60)
    print("  LangChain 多智能体协作系统 v2.0")
    print(f"  Session: {session_id}")
    if guard.enabled:
        print(f"  护栏: 已启用 ({len(guard.input_rules)} 输入规则 + {len(guard.output_rules)} 输出规则)")
    else:
        print("  护栏: 未启用")
    print("=" * 60)
    print("输入 exit 退出\n")

    while True:
        user_query = input(">>> ").strip()
        if user_query.lower() in ["exit", "quit", "q"]:
            break
        if not user_query:
            continue

        ok, cleaned, violations = guard.pre_process(user_query, session_id)
        if not ok:
            print(f"\n[Guardrails Blocked] {violations[0].get('message', '输入被拦截')}")
            continue

        print("\n--- 开始处理 ---\n")

        try:
            result = guard.wrap_graph({
                "query": cleaned,
                "session_id": session_id,
            })

            if result.get("_guardrails_blocked"):
                print(f"\n[Guardrails Blocked] {result.get('final_answer', '')}")
                continue

            print("\n" + "=" * 60)
            print("  [规划结果]")
            print(result.get("sub_tasks", "")[:300])
            print("\n" + "-" * 60)
            print("  [最终回答]")
            print(result.get("final_answer", "无输出"))
            print("=" * 60)

        except Exception as e:
            print(f"\n[Error] {type(e).__name__}: {e}")

    print("\n性能统计:")
    for name, stats in get_perf_stats().items():
        print(f"  {name}: {stats['count']}次, 平均{stats['avg_ms']}ms, 最大{stats['max_ms']}ms")
    print(f"Token 预估: {get_token_usage()}")


if __name__ == "__main__":
    main()
