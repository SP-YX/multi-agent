"""
技能系统（Skill System）— 为 Agent 提供模块化、可插拔的专业能力。

每个 Skill 是一个自包含的能力模块，具备：
  - 元数据（名称、描述、分类、触发关键词）
  - 标准化的 execute() 接口
  - YAML 配置驱动的注册与发现机制

使用方式：
  from skills.skill_registry import SkillRegistry
  registry = SkillRegistry()
  skills = registry.get_matching_skills(user_query)
  for skill in skills:
      result = skill.execute(input_text)

  # 或使用 SkillAgent 一站式调用
  from skills.skill_agent import SkillAgent
  agent = SkillAgent()
  answer = agent.run("今天天气怎么样？")
"""

from .base_skill import BaseSkill, SkillResult
from .skill_registry import SkillRegistry
from .skill_agent import SkillAgent

__all__ = [
    "BaseSkill",
    "SkillResult",
    "SkillRegistry",
    "SkillAgent",
]
