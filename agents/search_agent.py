"""
SearchAgent — 联网搜索智能体

职责：调用外部搜索引擎获取实时互联网信息。
补充 RAG 的不足（时效性、公开数据）。

工具依赖：web_search（Tavily / DuckDuckGo 双引擎）
中间件：监控、日志、性能、错误兜底
"""

from langchain.agents import create_agent
from agent_tools.web_search_tool import web_search_tool
from agent_tools.middleware import (
    monitor_tool,
    log_before_model,
    performance_middleware,
    error_handler_middleware,
)
from utils.prompts_tool import get_search_prompts
from .base_agent import BaseAgent


class SearchAgent(BaseAgent):
    """联网搜索智能体：实时从互联网获取公开信息。"""

    def __init__(self):
        """初始化：注入全局模型并创建 Agent（search_prompt + web_search）。"""
        super().__init__()
        self.agent = create_agent(
            model=self.model,
            system_prompt=get_search_prompts(),
            tools=[web_search_tool],
            middleware=[
                monitor_tool,
                log_before_model,
                performance_middleware,
                error_handler_middleware,
            ],
        )

    def run(self, user_input: str, **kwargs):
        """
        执行联网搜索。
        Args:
            user_input: 搜索查询（通常来自 plan 的子任务）
        Returns: 结构化的搜索结果文本
        """
        messages = []
        if self.memory_context:
            messages.append({"role": "system", "content": f"历史对话：\n{self.memory_context}"})
        messages.append({"role": "user", "content": user_input})
        input = {"messages": messages}
        result = self.agent.invoke(input)
        return result["messages"][-1].content
