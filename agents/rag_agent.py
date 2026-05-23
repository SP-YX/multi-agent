"""
RAGAgent — 知识库检索智能体

职责：基于 ChromaDB 向量库检索本地文档，生成带引用的回答。
适用于企业内部知识库、产品文档、技术手册等私有资料的问答。

工具依赖：rag_summarize（RAG 检索 + 生成）
中间件：监控、日志、性能、错误兜底
"""

from langchain.agents import create_agent
from agent_tools.middleware import (
    monitor_tool,
    log_before_model,
    performance_middleware,
    error_handler_middleware,
)
from agent_tools.rag_tool import rag_summarize
from utils.prompts_tool import get_rag_prompts
from .base_agent import BaseAgent


class RAGAgent(BaseAgent):
    """知识库检索智能体：从本地向量库检索并生成上下文相关回答。"""

    def __init__(self):
        """初始化：注入全局模型并创建 Agent（rag_prompt + rag_summarize）。"""
        super().__init__()
        self.agent = create_agent(
            model=self.model,
            system_prompt=get_rag_prompts(),
            tools=[rag_summarize],
            middleware=[
                monitor_tool,
                log_before_model,
                performance_middleware,
                error_handler_middleware,
            ],
        )

    def run(self, user_input: str, **kwargs):
        """
        执行 RAG 检索与生成。
        Args:
            user_input: 用户查询（通常来自 plan_agent 的子任务）
        Returns: 基于知识库的检索回答
        """
        input = {"messages": [{"role": "user", "content": user_input}]}
        result = self.agent.invoke(input)
        return result["messages"][-1].content
