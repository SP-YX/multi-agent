"""
CoderAgent — 代码生成与执行智能体

职责：根据需求自动生成 Python 代码，在安全沙箱中执行并返回结果。
适用于数据处理、计算、图表生成、自动化脚本等场景。

工具依赖：code_execute（AST 安全检查 + subprocess 沙箱）
中间件：监控、日志、性能、错误兜底
"""

from langchain.agents import create_agent
from agent_tools.code_interpreter import code_exec_tool
from agent_tools.middleware import (
    monitor_tool,
    log_before_model,
    performance_middleware,
    error_handler_middleware,
)
from utils.prompts_tool import get_coder_prompts
from .base_agent import BaseAgent


class CoderAgent(BaseAgent):
    """代码智能体：自动生成代码并在安全沙箱中执行。"""

    def __init__(self):
        """初始化：注入全局模型并创建 Agent（coder_prompt + code_execute）。"""
        super().__init__()
        self.agent = create_agent(
            model=self.model,
            system_prompt=get_coder_prompts(),
            tools=[code_exec_tool],
            middleware=[
                monitor_tool,
                log_before_model,
                performance_middleware,
                error_handler_middleware,
            ],
        )

    def run(self, user_input: str, **kwargs):
        """
        执行代码生成与运行。
        Args:
            user_input: 编程任务描述
        Returns: 代码执行结果文本
        """
        messages = []
        if self.memory_context:
            messages.append({"role": "system", "content": f"历史对话：\n{self.memory_context}"})
        messages.append({"role": "user", "content": user_input})
        input = {"messages": messages}
        result = self.agent.invoke(input)
        return result["messages"][-1].content
