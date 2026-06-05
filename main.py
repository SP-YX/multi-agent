"""
CLI 入口 — 多智能体系统的命令行交互界面。

通过标准输入读取用户任务，逐轮调用 LangGraph 流水线，
输出最终回答，并在退出时打印性能统计。

特殊命令：
  /skill create   创建自定义技能
  /skill list     列出所有技能
  /skill show     查看技能详情
  /skill delete   删除自定义技能
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


def _init_skill_registry():
    """初始化技能注册中心并返回。"""
    try:
        from skills.skill_registry import SkillRegistry
        reg = SkillRegistry()
        n = reg.discover_all()
        return reg, n
    except Exception:
        return None, 0


def _handle_skill_command(cmd: str, registry):
    """处理 /skill 开头的技能管理命令。"""
    from skills.dynamic_skill import DynamicSkill, create_dynamic_skill_from_form

    parts = cmd.split()
    sub = parts[1] if len(parts) > 1 else "help"

    if sub == "list":
        meta_list = registry.list_metadata()
        if not meta_list:
            print("  当前没有已注册的技能。")
            return
        print(f"\n  已注册技能 ({len(meta_list)} 个):")
        print(f"  {'名称':<20} {'分类':<12} {'状态':<8} 描述")
        print(f"  {'-'*20} {'-'*12} {'-'*8} {'-'*40}")
        for m in meta_list:
            status = "启用" if m["enabled"] else "禁用"
            print(f"  {m['name']:<20} {m['category']:<12} {status:<8} {m['description'][:40]}")

    elif sub == "show":
        if len(parts) < 3:
            print("  用法: /skill show <技能名称>")
            return
        name = parts[2]
        skill = registry.get(name)
        if not skill:
            print(f"  技能 '{name}' 未找到。")
            return
        meta = skill.get_metadata()
        print(f"\n  [技能详情] {meta['name']}")
        print(f"  描述: {meta['description']}")
        print(f"  分类: {meta['category']}")
        print(f"  关键词: {', '.join(meta['keywords'])}")
        print(f"  启用: {meta['enabled']}")
        if hasattr(skill, 'system_prompt'):
            print(f"  提示词: {skill.system_prompt[:200]}")
        if hasattr(skill, 'tool_names') and skill.tool_names:
            print(f"  工具: {', '.join(skill.tool_names)}")

    elif sub == "create":
        print("\n  [创建自定义技能]")
        print("  (直接回车取消)\n")
        name = input("  技能名称(英文): ").strip()
        if not name:
            return
        desc = input("  技能描述(一句话): ").strip()
        if not desc:
            return
        kw_raw = input("  触发关键词(逗号分隔): ").strip()
        keywords = [k.strip() for k in kw_raw.split(",") if k.strip()]
        print("  (提示词中可用 {input} 代表用户输入)")
        prompt = input("  System Prompt: ").strip()
        if not prompt:
            return
        tools_raw = input("  绑定工具(可选,逗号分隔,可用: web_search_tool,code_execute等): ").strip()
        tools = [t.strip() for t in tools_raw.split(",") if t.strip()]

        skill = create_dynamic_skill_from_form(name, desc, keywords, prompt, tools)
        ok = registry.add_user_skill(skill)
        if ok:
            # 刷新路由器的技能缓存
            try:
                from agents.router_agent import router_agent
                router_agent._refresh_skill_meta()
            except Exception:
                pass
            print(f"  ✔ 技能 '{name}' 创建成功！输入 '{name}' 相关的问题即可触发。")
        else:
            print(f"  ✘ 技能 '{name}' 创建失败。")

    elif sub == "delete":
        if len(parts) < 3:
            print("  用法: /skill delete <技能名称>")
            return
        name = parts[2]
        confirm = input(f"  确认删除技能 '{name}'? (y/N): ").strip().lower()
        if confirm == "y":
            ok = registry.remove_user_skill(name)
            if ok:
                try:
                    from agents.router_agent import router_agent
                    router_agent._refresh_skill_meta()
                except Exception:
                    pass
                print(f"  ✔ 技能 '{name}' 已删除。")
            else:
                print(f"  ✘ 技能 '{name}' 删除失败。")
        else:
            print("  已取消。")

    else:
        print("  /skill 命令:")
        print("    /skill create         创建自定义技能")
        print("    /skill list           列出所有技能")
        print("    /skill show <name>    查看技能详情")
        print("    /skill delete <name>  删除自定义技能")


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

    registry, n = _init_skill_registry()
    if n:
        print(f"  技能: {n} 个已加载（含自定义）")

    print("=" * 60)
    print("输入 exit 退出 | /skill 管理自定义技能\n")

    while True:
        user_query = input(">>> ").strip()
        if user_query.lower() in ["exit", "quit", "q"]:
            break
        if not user_query:
            continue

        # 技能管理命令
        if user_query.startswith("/skill"):
            if registry:
                _handle_skill_command(user_query, registry)
            else:
                print("  技能系统未加载。")
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
            if result.get("sub_tasks"):
                print("  [规划结果]")
                print(result.get("sub_tasks")[:300])
                print("-" * 60)
            if result.get("skill_result"):
                print("  [技能执行结果]")
                print(result.get("skill_result")[:500])
                print("-" * 60)
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
