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
        self._skill_meta_cache = ""
        self._refresh_skill_meta()
        self.chain = ChatPromptTemplate.from_template(get_router_prompts()) | self.model

    def _refresh_skill_meta(self):
        """从 SkillRegistry 拉取可用技能元数据，注入路由提示词。"""
        try:
            from skills.skill_registry import SkillRegistry
            registry = SkillRegistry()
            registry.discover_all()
            meta_list = registry.list_metadata()
            if meta_list:
                lines = []
                for m in meta_list:
                    kw = ", ".join(m["keywords"][:4])
                    lines.append(f"  - {m['name']}: {m['description']}（触发词: {kw}）")
                self._skill_meta_cache = "\n".join(lines)
            else:
                self._skill_meta_cache = "（无可用技能）"
        except Exception:
            self._skill_meta_cache = "（技能系统未加载）"

    def run(self, user_input: str, **kwargs) -> str:
        result = self.chain.invoke({
            "query": user_input,
            "available_skills": self._skill_meta_cache,
        })
        return result.content.strip()
