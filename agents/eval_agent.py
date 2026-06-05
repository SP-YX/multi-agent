"""
EvalAgent — 评估与校验智能体

职责：对 Summary Agent 生成的最终回答进行多维度质量评估，
提供评分、改进建议，并判断是否需要重新生成。

评估维度：
  - completeness（完整度）
  - accuracy（准确度）
  - clarity（清晰度）
  - relevance（相关度）
  - safety（安全度）

无工具依赖，纯 LLM 调用。
"""

import json
import logging
from typing import Optional

from langchain.agents import create_agent
from agent_tools.middleware import (
    monitor_tool,
    log_before_model,
    performance_middleware,
    token_counter_middleware,
)
from utils.prompts_tool import get_eval_prompts
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class EvalAgent(BaseAgent):
    """评估与校验智能体：对 Agent 输出进行质量评分与改进建议。"""

    def __init__(self):
        """初始化：注入全局模型并创建 Agent（eval_prompt，无工具）。"""
        super().__init__()
        self.agent = create_agent(
            model=self.model,
            system_prompt=get_eval_prompts(),
            tools=[],
            middleware=[
                monitor_tool,
                log_before_model,
                performance_middleware,
                token_counter_middleware,
            ],
        )
        self._load_config()

    def _load_config(self):
        """加载评估配置（阈值等）。"""
        try:
            import yaml
            from utils.config_tool import get_abs_path
            path = get_abs_path("config/eval_config.yml")
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            cfg = raw.get("eval", {})
            self.enabled = cfg.get("enabled", True)
            self.pass_threshold = cfg.get("thresholds", {}).get("pass", 7.0)
            self.warn_threshold = cfg.get("thresholds", {}).get("warn", 5.0)
            self.fail_threshold = cfg.get("thresholds", {}).get("fail", 3.0)
            self.max_retries = cfg.get("on_low_score", {}).get("max_retries", 1)
        except Exception as e:
            logger.warning(f"EvalAgent 配置加载失败，使用默认值: {e}")
            self.enabled = True
            self.pass_threshold = 7.0
            self.warn_threshold = 5.0
            self.fail_threshold = 3.0
            self.max_retries = 1

    def run(self, user_input: str = "", **kwargs) -> str:
        """
        执行评估。
        Args:
            user_input: 原始用户查询
            **kwargs: 需要包含 final_answer
        Returns: JSON 格式的评估报告
        """
        final_answer = kwargs.get("final_answer", "")

        if not final_answer:
            return json.dumps({
                "scores": {"completeness": 0, "accuracy": 0, "clarity": 0, "relevance": 0, "safety": 0},
                "weighted_total": 0.0,
                "passed": False,
                "summary": "无可用的最终回答进行评估",
                "strengths": [],
                "weaknesses": ["最终回答为空"],
                "suggestions": ["请先完成最终回答的生成"],
            }, ensure_ascii=False)

        combined_input = f"""## Original Query
{user_input}

## Final Answer
{final_answer}

Please evaluate the final answer above."""
        messages = []
        if self.memory_context:
            messages.append({"role": "system", "content": f"历史对话：\n{self.memory_context}"})
        messages.append({"role": "user", "content": combined_input})
        input_data = {"messages": messages}
        result = self.agent.invoke(input_data)
        content = result["messages"][-1].content

        return content

    def evaluate(self, query: str, final_answer: str) -> dict:
        """
        方便的评估接口：直接调用 run() 并解析 JSON 结果。
        Args:
            query: 用户原始查询
            final_answer: 待评估的最终回答
        Returns: 解析后的评估报告 dict
        """
        raw = self.run(query, final_answer=final_answer)
        return self._parse_result(raw)

    def _parse_result(self, raw: str) -> dict:
        """
        解析 LLM 返回的评估报告 JSON。
        Args:
            raw: LLM 返回的原始文本
        Returns: 解析后的字典
        """
        try:
            # 尝试从 ```json ... ``` 代码块中提取
            if "```json" in raw:
                start = raw.index("```json") + 7
                end = raw.index("```", start) if "```" in raw[start:] else len(raw)
                json_str = raw[start:end].strip()
            elif "```" in raw:
                start = raw.index("```") + 3
                end = raw.index("```", start) if "```" in raw[start:] else len(raw)
                json_str = raw[start:end].strip()
            else:
                json_str = raw.strip()

            report = json.loads(json_str)
            # 确保关键字段存在
            report.setdefault("scores", {})
            report.setdefault("weighted_total", 0.0)
            report.setdefault("passed", False)
            report.setdefault("summary", "")
            report.setdefault("strengths", [])
            report.setdefault("weaknesses", [])
            report.setdefault("suggestions", [])
            return report
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"EvalAgent JSON 解析失败: {e}")
            return {
                "scores": {},
                "weighted_total": 0.0,
                "passed": False,
                "summary": f"评估报告解析失败: {e}",
                "strengths": [],
                "weaknesses": ["无法解析评估结果"],
                "suggestions": ["请人工复核"],
                "_raw": raw,
            }

    def needs_retry(self, report: dict) -> bool:
        """
        判断是否需要重新生成回答。
        Args:
            report: evaluate() 返回的评估报告
        Returns: 是否需要重试
        """
        weighted = report.get("weighted_total", 0.0)
        return weighted < self.fail_threshold
