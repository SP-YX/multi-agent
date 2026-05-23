"""
SummaryAgent — 结果汇总智能体

职责：汇聚全流程中间结果（规划 + RAG + 搜索 + 代码），
合成结构化的最终回答。是整个流水线的最后一环。

无工具依赖，纯 LLM 调用。
输入通过 **kwargs 接收其他 Agent 的输出。
"""

from langchain.agents import create_agent
from agent_tools.middleware import (
    monitor_tool,
    log_before_model,
    performance_middleware,
    memory_inject_middleware,
    token_counter_middleware,
)
from utils.prompts_tool import get_summary_prompts
from .base_agent import BaseAgent


class SummaryAgent(BaseAgent):
    """结果汇总智能体：将多源结果整合为统一、完整的最终答案。"""

    def __init__(self):
        """初始化：注入全局模型并创建 Agent（summary_prompt，无工具）。"""
        super().__init__()
        self.agent = create_agent(
            model=self.model,
            system_prompt=get_summary_prompts(),
            tools=[],
            middleware=[
                monitor_tool,
                log_before_model,
                performance_middleware,
                memory_inject_middleware,
                token_counter_middleware,
            ],
        )

    def run(self, user_input: str = "", **kwargs):
        """
        执行结果汇总。
        Args:
            user_input: 原始用户查询
            **kwargs: 需要包含 sub_tasks / rag_res / search_res / code_res
        Returns: 整合后的最终回答
        """
        # 从 kwargs 中提取各 Agent 的中间结果
        rag_res = kwargs.get("rag_res", "")
        search_res = kwargs.get("search_res", "")
        code_res = kwargs.get("code_res", "")
        plan_res = kwargs.get("sub_tasks", "")

        # 将所有中间结果拼接到一个输入中，交给 LLM 统一汇总
        combined_input = f"""## Original Query
{user_input}

## Task Plan
{plan_res}

## Knowledge Base Results (RAG)
{rag_res}

## Web Search Results
{search_res}

## Code Execution Results
{code_res}

Please synthesize all the above information into a comprehensive final answer.
"""
        input = {"messages": [{"role": "user", "content": combined_input}]}
        result = self.agent.invoke(input)
        return result["messages"][-1].content
