"""
EvalAgent 单元测试

运行方式：python test_eval_agent.py

测试覆盖：
  - EvalAgent 初始化与配置加载
  - 评估报告 JSON 解析（直接 JSON / 代码块包裹）
  - 阈值判断逻辑（pass / warn / fail）
  - 边界情况处理（空输入、非法 JSON）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.eval_agent import EvalAgent


def test_initialization():
    """测试 EvalAgent 初始化和配置加载"""
    agent = EvalAgent()
    assert agent.enabled == True
    assert agent.pass_threshold == 7.0
    assert agent.warn_threshold == 5.0
    assert agent.fail_threshold == 3.0
    assert agent.max_retries == 1
    assert hasattr(agent, 'model')
    assert hasattr(agent, 'agent')
    print("[PASS] test_initialization")


def test_parse_direct_json():
    """测试直接 JSON 解析"""
    agent = EvalAgent()
    raw = '{"scores": {"completeness": 8, "accuracy": 7}, "weighted_total": 8.25, "passed": true, "summary": "Good", "strengths": ["Clear"], "weaknesses": [], "suggestions": []}'
    result = agent._parse_result(raw)
    assert result["passed"] == True
    assert result["weighted_total"] == 8.25
    assert result["summary"] == "Good"
    assert "Clear" in result["strengths"]
    print("[PASS] test_parse_direct_json")


def test_parse_json_block():
    """测试 ```json 代码块解析"""
    agent = EvalAgent()
    raw = '```json\n{"weighted_total": 6.5, "passed": false, "summary": "Needs work", "strengths": [], "weaknesses": ["Incomplete"], "suggestions": ["Add details"]}\n```'
    result = agent._parse_result(raw)
    assert result["passed"] == False
    assert result["weighted_total"] == 6.5
    assert "Incomplete" in result["weaknesses"]
    print("[PASS] test_parse_json_block")


def test_parse_with_defaults():
    """测试解析结果自动补齐缺失字段"""
    agent = EvalAgent()
    raw = '{"weighted_total": 5.0}'
    result = agent._parse_result(raw)
    assert result["weighted_total"] == 5.0
    assert result["passed"] == False
    assert result["scores"] == {}
    assert result["strengths"] == []
    assert result["weaknesses"] == []
    assert result["suggestions"] == []
    print("[PASS] test_parse_with_defaults")


def test_parse_invalid_json():
    """测试非法 JSON 容错"""
    agent = EvalAgent()
    raw = "这不是 JSON"
    result = agent._parse_result(raw)
    assert result["weighted_total"] == 0.0
    assert result["passed"] == False
    assert "_raw" in result
    print("[PASS] test_parse_invalid_json")


def test_needs_retry():
    """测试重试判断逻辑"""
    agent = EvalAgent()
    assert agent.needs_retry({"weighted_total": 2.0}) == True
    assert agent.needs_retry({"weighted_total": 3.0}) == False
    assert agent.needs_retry({"weighted_total": 6.0}) == False
    assert agent.needs_retry({"weighted_total": 10.0}) == False
    print("[PASS] test_needs_retry")


def test_empty_final_answer():
    """测试空 final_answer 处理"""
    agent = EvalAgent()
    result = agent.run("测试查询", final_answer="")
    import json
    parsed = json.loads(result)
    assert parsed["weighted_total"] == 0.0
    assert parsed["passed"] == False
    assert "最终回答为空" in parsed["weaknesses"]
    print("[PASS] test_empty_final_answer")


def test_evaluate_interface():
    """测试 evaluate() 便捷接口"""
    agent = EvalAgent()
    result = agent.evaluate("测试查询", "")
    assert result["weighted_total"] == 0.0
    assert result["passed"] == False
    print("[PASS] test_evaluate_interface")


if __name__ == "__main__":
    test_initialization()
    test_parse_direct_json()
    test_parse_json_block()
    test_parse_with_defaults()
    test_parse_invalid_json()
    test_needs_retry()
    test_empty_final_answer()
    test_evaluate_interface()
    print("\n所有 EvalAgent 测试通过!")
