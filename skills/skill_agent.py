"""
SkillAgent — 技能调度智能体。

职责：接收用户输入 → SkillRegistry 匹配技能 → 调用技能 → 整合结果 → 返回回答。

与普通 Agent 的区别：
  - 不依赖 LangChain create_agent，而是直接调用 SkillRegistry
  - 轻量级，一次 LLM 调用即可确定技能选择和结果整合
  - 可独立运行，也可作为 LangGraph 的一个节点
"""

from typing import Any
from skills.skill_registry import SkillRegistry
from utils.prompts_tool import get_skill_prompts


class SkillAgent:
    """技能调度智能体 — 负责技能的路由、执行和结果整合。"""

    def __init__(self):
        self.registry = SkillRegistry()
        self.registry.discover_all()
        self.available_skills_meta = ""
        self.memory_context = ""
        self._refresh_meta()

    def set_memory_context(self, context: str):
        """注入会话记忆上下文（兼容 BaseAgent 接口）。"""
        self.memory_context = context

    def _refresh_meta(self):
        """刷新可用技能元数据文本（供 LLM 参考）。"""
        meta_list = self.registry.list_metadata()
        if not meta_list:
            self.available_skills_meta = "当前没有可用技能。"
            return

        lines = ["| 技能名称 | 描述 | 关键词 |"]
        lines.append("|---------|------|--------|")
        for m in meta_list:
            kw = ", ".join(m["keywords"][:5])
            lines.append(f"| {m['name']} | {m['description']} | {kw} |")
        self.available_skills_meta = "\n".join(lines)

    def run(self, user_input: str, **kwargs) -> str:
        """主入口：分析输入 → 匹配技能 → 执行 → 整合结果。

        Args:
            user_input: 用户查询
            **kwargs: 额外参数（传递给技能）

        Returns:
            整合后的回答文本
        """
        if not user_input.strip():
            return "请输入有效的问题。"

        # 1. 匹配技能（先用关键词快速匹配）
        matched_skills = self.registry.get_matching_skills(user_input)

        if not matched_skills:
            # 没有匹配到技能 → 直接 LLM 回答
            return self._direct_answer(user_input)

        # 2. 如果匹配到多个技能，用 LLM 选择最合适的
        if len(matched_skills) > 1:
            selected = self._llm_select_skill(user_input, matched_skills)
        else:
            selected = matched_skills

        # 3. 执行技能
        results = []
        for skill in selected:
            try:
                result = skill.execute(input=user_input, **kwargs)
                results.append((skill.name, result))
            except Exception as e:
                results.append((skill.name, f"[执行错误] {e}"))

        # 4. 整合结果
        return self._integrate_results(user_input, results)

    def _direct_answer(self, query: str) -> str:
        """没有技能匹配时，直接 LLM 回答。"""
        from models.my_model import chat_model
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = []
        if self.memory_context:
            messages.append(SystemMessage(content=f"历史对话：\n{self.memory_context}"))
        messages.append(SystemMessage(content="你是一个友好的AI助手。用户的问题不需要调用专业技能，请直接回答。"))
        messages.append(HumanMessage(content=query))
        result = chat_model.invoke(messages)
        return result.content

    def _llm_select_skill(self, query: str, skills: list) -> list:
        """用 LLM 从多个候选技能中选择最合适的。

        Args:
            query: 用户查询
            skills: 候选技能列表

        Returns:
            选中的技能列表（通常 1-2 个）
        """
        from models.my_model import chat_model
        from langchain_core.prompts import ChatPromptTemplate

        candidates = "\n".join(
            f"- {s.name}: {s.description}" for s in skills
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "根据用户查询，从候选中选择最合适的技能。只返回技能名称，用逗号分隔。"),
            ("human", "候选技能：\n{candidates}\n\n用户查询：{query}"),
        ])

        result = chat_model.invoke(prompt.format(candidates=candidates, query=query))
        selected_names = [n.strip() for n in result.content.split(",") if n.strip()]

        return [s for s in skills if s.name in selected_names] or [skills[0]]

    def _integrate_results(self, query: str, results: list[tuple[str, Any]]) -> str:
        """将技能执行结果整合为回答文本。

        Args:
            query: 用户原始查询
            results: [(技能名称, 执行结果), ...]

        Returns:
            整合后的回答
        """
        from models.my_model import chat_model
        from langchain_core.messages import HumanMessage, SystemMessage

        skill_outputs = "\n\n".join(
            f"## [{name}]\n{result}" for name, result in results
        )

        system_content = get_skill_prompts().replace("{available_skills}", self.available_skills_meta)

        messages = []
        if self.memory_context:
            messages.append(SystemMessage(content=f"历史对话：\n{self.memory_context}"))
        messages.append(SystemMessage(content=system_content))
        messages.append(HumanMessage(content=f"用户问题：{query}\n\n技能执行结果：\n{skill_outputs}"))

        result = chat_model.invoke(messages)
        return result.content
