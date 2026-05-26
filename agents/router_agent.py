"""
RouterAgent — 路由智能体（轻量版）

职责：用一次直接 LLM 调用对用户问题进行快速分类，
避免所有问题都走全量 Agent 流水线。
"""

from langchain_core.prompts import ChatPromptTemplate
from .base_agent import BaseAgent
from utils.prompts_tool import get_router_prompts

class RouterAgent(BaseAgent):
    """路由智能体（轻量版）"""

    def __init__(self):
        super().__init__()
        prompt_template = ChatPromptTemplate.from_template(get_router_prompts())
        self.chain = prompt_template | self.model

    def run(self, user_input: str, **kwargs) -> str:
        result = self.chain.invoke({"query": user_input})
        return result.content.strip()
