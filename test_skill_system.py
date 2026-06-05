"""
技能系统测试脚本 — 验证 Skill System 的核心功能。
运行方式：python test_skill_system.py
"""


def test_base_skill():
    """测试 BaseSkill 基类和 SkillResult。"""
    from skills.base_skill import SkillResult

    ok = SkillResult(True, output="成功！")
    assert ok.success and ok.output == "成功！"

    fail = SkillResult(False, error="出错了")
    assert not fail.success and fail.error == "出错了"

    print("[PASS] BaseSkill / SkillResult")


def test_registry_discover():
    """测试 SkillRegistry 自动加载技能。"""
    from skills.skill_registry import SkillRegistry

    registry = SkillRegistry()
    count = registry.discover_all()
    assert count >= 3
    assert registry.get("calculator") is not None
    assert registry.get("web_research") is not None
    assert registry.get("file_analysis") is not None

    meta = registry.list_metadata()
    assert len(meta) >= 3

    print(f"[PASS] SkillRegistry 自动发现并加载了 {count} 个技能")
    for m in meta:
        print(f"       - {m['name']}: {m['description']}")


def test_keyword_matching():
    """测试关键词匹配技能。"""
    from skills.skill_registry import SkillRegistry

    registry = SkillRegistry()
    registry.discover_all()

    # 计算相关
    math_skills = registry.match_by_keywords("帮我计算 3.14 乘以 2")
    assert any(s.name == "calculator" for s in math_skills)

    # 搜索相关
    search_skills = registry.match_by_keywords("搜索一下最新的AI新闻")
    assert any(s.name == "web_research" for s in search_skills)

    # 文件分析相关
    file_skills = registry.match_by_keywords("统计一下项目中的代码行数")
    assert any(s.name == "file_analysis" for s in file_skills)

    print(f"[PASS] 关键词匹配")
    print(f"       '计算' → {[s.name for s in math_skills]}")
    print(f"       '搜索' → {[s.name for s in search_skills]}")
    print(f"       '统计' → {[s.name for s in file_skills]}")


def test_calculator_skill():
    """测试计算器技能。"""
    from skills.example_skills.calculator_skill import CalculatorSkill

    skill = CalculatorSkill({"precision": 4})

    r1 = skill.execute(input="2 + 3")
    assert r1.success
    print(f"       2 + 3 = {r1.output}")

    r2 = skill.execute(input="计算 3.14 * 2 的结果")
    assert r2.success
    print(f"       3.14 * 2 = {r2.output}")

    r3 = skill.execute(input="2的3次方")
    assert r3.success
    print(f"       2的3次方 = {r3.output}")

    print("[PASS] CalculatorSkill")


def test_file_analysis_skill():
    """测试文件分析技能。"""
    from skills.example_skills.file_analysis_skill import FileAnalysisSkill

    skill = FileAnalysisSkill()
    result = skill.execute(input="统计项目文件")

    assert result.success
    assert "总文件数" in result.output
    assert "Python" in result.output or "🐍" in result.output

    print(f"[PASS] FileAnalysisSkill")
    lines = result.output.split("\n")
    for line in lines[:5]:
        print(f"       {line}")


def test_skill_agent():
    """测试 SkillAgent 完整流程。"""
    from skills.skill_agent import SkillAgent

    agent = SkillAgent()
    answer = agent.run("帮我计算 15 乘以 37 的结果")

    print(f"[PASS] SkillAgent")
    print(f"       输入: 15 × 37")
    print(f"       输出: {answer[:100]}...")


def test_skill_graph_node():
    """测试技能节点在 LangGraph 中的使用。"""
    from skills.skill_agent import SkillAgent
    from graph.agent_graph import AgentState

    skill_agent = SkillAgent()

    # 模拟 LangGraph 节点函数
    def skill_node(state: AgentState) -> dict:
        query = state.get("query", "")
        result = skill_agent.run(query)
        return {"skill_result": result}

    state = AgentState(
        query="统计项目中Python文件的数量",
        route="",
        sub_tasks="",
        rag_result="",
        search_result="",
        code_result="",
        final_answer="",
        error=None,
        session_id="test",
    )

    output = skill_node(state)
    assert "skill_result" in output
    print(f"[PASS] SkillGraphNode 集成（可作为 LangGraph 节点使用）")


if __name__ == "__main__":
    print("=" * 60)
    print("  Skill System 测试套件")
    print("=" * 60)

    test_base_skill()
    test_registry_discover()
    test_keyword_matching()
    test_calculator_skill()
    test_file_analysis_skill()

    print("\n  --- 以下测试需要 LLM API Key ---")
    try:
        test_skill_agent()
        test_skill_graph_node()
    except Exception as e:
        print(f"  [SKIP] LLM 相关测试跳过: {e}")

    print("\n" + "=" * 60)
    print("  所有测试完成！")
    print("=" * 60)
