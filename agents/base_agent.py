"""
Agent 抽象基类 — 所有智能体的统一接口规范。

子类必须实现 __init__()（调用 super().__init__() 后初始化 self.agent）
和 run()。

所有 Agent 共享同一个全局 chat_model（DashScope Tongyi 实例）。
"""

from abc import ABC, abstractmethod
from models.my_model import chat_model


class BaseAgent(ABC):
    """所有智能体的抽象基类，定义统一的生命周期。"""

    def __init__(self):
        """初始化：注入全局模型实例。"""
        self.model = chat_model       # 共享的 LLM 实例
        self.agent = None             # 由子类 __init__ 中创建
        self.memory_context = ""      # 会话记忆上下文

    def set_memory_context(self, context: str):
        """注入会话记忆上下文。"""
        self.memory_context = context

    @abstractmethod
    def run(self, user_input: str, **kwargs):
        """
        执行 Agent 的核心入口。
        Args:
            user_input: 用户输入的查询字符串
            **kwargs: 额外参数（用于 summary_agent 传递多源结果）
        Returns: Agent 的完整输出文本
        """
        pass
