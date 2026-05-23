"""
PlanAgent — 任务规划智能体

职责：接收原始用户需求，拆解为结构化任务计划。
输出包含：需求分析、技术选型、模块划分、执行步骤、风险评估。

无工具依赖，完全依赖 LLM 的推理能力。
使用中间件：监控、日志、性能、记忆注入、Token 统计。
"""

from langchain.agents import create_agent
from agent_tools.middleware import (
    monitor_tool,
    log_before_model,
    performance_middleware,
    memory_inject_middleware,
    token_counter_middleware,
)
from utils.prompts_tool import get_plan_prompts
from .base_agent import BaseAgent


class PlanAgent(BaseAgent):
    """任务规划智能体：将用户需求拆解为可执行的子任务计划。"""

    def __init__(self):
        """初始化：注入全局模型并创建 Agent（plan_prompt，无工具）。"""
        super().__init__()
        self.agent = create_agent(
            model=self.model,
            system_prompt=get_plan_prompts(),
            tools=[],
            middleware=[
                monitor_tool,
                log_before_model,
                performance_middleware,
                memory_inject_middleware,
                token_counter_middleware,
            ],
        )

    def run(self, user_input: str, **kwargs):
        """
        执行任务规划。
        Args:
            user_input: 用户原始需求
        Returns: 结构化的任务计划文本
        """
        input = {"messages": [{"role": "user", "content": user_input}]}
        result = self.agent.invoke(input)
        return result["messages"][-1].content
