"""
技能注册中心 — 负责技能的动态发现、加载、匹配与生命周期管理。

核心能力：
  1. YAML 配置驱动 → 自动加载所有已启用的技能
  2. 关键词匹配 → 根据用户查询快速筛选合适技能
  3. LLM 路由 → 复杂场景用 LLM 判断最合适技能
  4. 技能链 → 多技能顺序执行

用法示例：
  registry = SkillRegistry()
  registry.discover_all()

  # 关键词匹配
  skills = registry.match_by_keywords("今天天气怎么样")
  for skill in skills:
      result = skill.execute(location="北京")

  # 获取技能元数据列表（供 LLM 选择）
  meta = registry.list_metadata()
"""

import os
import importlib
from typing import Any
from skills.base_skill import BaseSkill
from utils.log_tool import logger
from utils.path_tool import get_abs_path

class SkillRegistry:
    """Skill注册中心,管理所有Skill实例。"""

    _instance = None
    _skills: dict[str, BaseSkill] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: str | None = None):
        if not hasattr(self, "_initialized"):
            self._config_path = config_path or self._default_config_path()
            self._skills = {}
            self._initialized = True

    @staticmethod
    def _default_config_path() -> str:
        return get_abs_path("config/skills.yml")

    # 技能注册
    def register(self, skill: BaseSkill) -> None:
        """手动注册一个技能实例。"""
        if not isinstance(skill, BaseSkill):
            raise TypeError(f"{type(skill).__name__} 不是 BaseSkill 实例")
        if not skill.name:
            raise ValueError("Skill名词不能为空!")
        self._skills[skill.name] = skill
        logger.info(f"[SkillRegistry] Skill: {skill.name} 注册成功!")

    def unregister(self, name: str) -> None:
        """注销指定技能。"""
        self._skills.pop(name, None)
        logger.info(f"[SkillRegistry] 注销技能: {name}")

    def get(self, name: str) -> BaseSkill | None:
        """按名称获取技能实例。"""
        return self._skills.get(name)

    def all(self) -> list[BaseSkill]:
        """获取所有已注册的技能列表。"""
        return list(self._skills.values())

    def list_metadata(self) -> list[dict[str, Any]]:
        """获取所有技能的元数据（供 LLM 选择技能时参考）。"""
        return [s.get_metadata() for s in self._skills.values() if s.enabled]

    # ── 技能匹配 API ─────────────────────────────────────

    def match_by_keywords(self, query: str) -> list[BaseSkill]:
        """基于关键词匹配技能（快速预过滤）。

        将用户查询与每个技能的 keywords 列表进行匹配，
        返回匹配上关键词的技能（按匹配数量降序）。
        """
        query_lower = query.lower()
        scored: list[tuple[int, BaseSkill]] = []

        for skill in self._skills.values():
            if not skill.enabled:
                continue
            match_count = sum(
                1 for kw in skill.keywords if kw.lower() in query_lower
            )
            if match_count > 0:
                scored.append((match_count, skill))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored]

    def match_by_llm(self, query: str, top_k: int = 3) -> list[BaseSkill]:
        """用 LLM 判断最合适的技能（精确匹配）。

        将技能元数据列表和用户查询一起发给 LLM，
        让 LLM 选择最相关的技能。
        """
        from models.my_model import chat_model
        from langchain_core.prompts import ChatPromptTemplate

        meta_list = self.list_metadata()
        if not meta_list:
            return []

        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个技能路由专家。根据用户查询，从以下技能中选择最合适的技能。"
                       "只返回技能名称，用逗号分隔，不要其他内容。"),
            ("human", "可用技能：\n{metadata}\n\n用户查询：{query}"),
        ])

        meta_str = "\n".join(
            f"- {m['name']}: {m['description']} (关键词: {', '.join(m['keywords'])})"
            for m in meta_list
        )

        result = chat_model.invoke(prompt.format(metadata=meta_str, query=query))
        selected = [name.strip() for name in result.content.split(",") if name.strip()]

        matched = []
        for name in selected[:top_k]:
            skill = self._skills.get(name)
            if skill and skill.enabled:
                matched.append(skill)

        return matched

    def get_matching_skills(self, query: str, use_llm: bool = False) -> list[BaseSkill]:
        """综合匹配：先用关键词过滤，再用 LLM 精排（可选）。

        Args:
            query: 用户查询
            use_llm: 是否启用 LLM 精排（默认只用关键词匹配）

        Returns:
            匹配的技能列表（按相关度降序）
        """
        keyword_matched = self.match_by_keywords(query)

        if not use_llm or not keyword_matched:
            return keyword_matched

        llm_matched = self.match_by_llm(query)
        seen = set()
        combined = []
        for skill in llm_matched + keyword_matched:
            if skill.name not in seen:
                seen.add(skill.name)
                combined.append(skill)
        return combined

    # ── 技能执行 API ─────────────────────────────────────

    def execute_skill(self, name: str, **kwargs) -> Any:
        """执行指定名称的技能。"""
        skill = self.get(name)
        if not skill:
            return f"[SkillRegistry] 技能 '{name}' 未注册"
        if not skill.enabled:
            return f"[SkillRegistry] 技能 '{name}' 已禁用"
        return skill.execute(**kwargs)

    def execute_chain(self, query: str, **kwargs) -> list[tuple[str, Any]]:
        """顺序执行所有匹配的技能，返回 (技能名称, 结果) 列表。"""
        skills = self.get_matching_skills(query)
        results = []
        for skill in skills:
            result = skill.execute(**kwargs)
            results.append((skill.name, result))
        return results

    # ── 用户自定义技能管理 ──────────────────────────────

    def add_user_skill(self, skill: BaseSkill) -> bool:
        """添加用户自定义技能并持久化到 user_skills.yml。"""
        self.register(skill)
        try:
            self._save_user_skills()
            return True
        except Exception as e:
            logger.error(f"[SkillRegistry] 保存用户技能失败: {e}")
            return False

    def remove_user_skill(self, name: str) -> bool:
        """删除用户自定义技能。"""
        self.unregister(name)
        try:
            self._save_user_skills()
            return True
        except Exception as e:
            logger.error(f"[SkillRegistry] 删除用户技能失败: {e}")
            return False

    def _get_user_skills_path(self) -> str:
        from utils.path_tool import get_abs_path
        return get_abs_path("config/user_skills.yml")

    def _save_user_skills(self):
        """将当前注册的所有 dynamic 类型技能回写到 user_skills.yml。"""
        import yaml

        skills_data = {}
        for name, skill in self._skills.items():
            if not hasattr(skill, 'system_prompt'):
                continue
            skills_data[name] = {
                "enabled": skill.enabled,
                "type": "dynamic",
                "description": skill.description,
                "keywords": skill.keywords,
                "category": getattr(skill, "category", "custom"),
                "system_prompt": skill.system_prompt,
                "tools": getattr(skill, "tool_names", []),
                "config": getattr(skill, "config", {}),
            }

        path = self._get_user_skills_path()
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump({"skills": skills_data}, f, allow_unicode=True, indent=2)

        logger.info(f"[SkillRegistry] 用户技能已保存: {len(skills_data)} 个")

    def _load_user_skills(self) -> int:
        """加载 config/user_skills.yml 中的用户自定义技能。"""
        import yaml
        from skills.dynamic_skill import DynamicSkill

        path = self._get_user_skills_path()
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except FileNotFoundError:
            return 0

        skills_cfg = raw.get("skills", {})
        count = 0
        for skill_name, cfg in skills_cfg.items():
            if not cfg.get("enabled", True):
                continue
            if cfg.get("type") == "dynamic":
                config = {
                    "name": skill_name,
                    "description": cfg.get("description", ""),
                    "keywords": cfg.get("keywords", []),
                    "category": cfg.get("category", "custom"),
                    "system_prompt": cfg.get("system_prompt", ""),
                    "tools": cfg.get("tools", []),
                    "enabled": True,
                }
                config.update(cfg.get("config", {}))
                instance = DynamicSkill(config)
                self.register(instance)
                count += 1

        if count:
            logger.info(f"[SkillRegistry] 用户自定义技能: {count} 个")
        return count

    # ── 技能发现与加载 ───────────────────────────────────

    def discover_all(self) -> int:
        """自动发现并注册所有已配置的技能。

        读取 config/skills.yml，根据配置动态导入并实例化技能类。
        返回成功加载的技能数量。
        """
        import yaml

        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"[SkillRegistry] 配置文件不存在: {self._config_path}")
            return 0

        skills_cfg = raw.get("skills", {})
        if not skills_cfg:
            logger.warning("[SkillRegistry] skills.yml 中未找到 skills 配置")
            return 0

        count = 0
        for skill_name, cfg in skills_cfg.items():
            if not cfg.get("enabled", True):
                logger.info(f"[SkillRegistry] 跳过已禁用技能: {skill_name}")
                continue

            module_path = cfg.get("module")
            class_name = cfg.get("class")
            if not module_path or not class_name:
                logger.warning(f"[SkillRegistry] 技能 '{skill_name}' 缺少 module/class 配置")
                continue

            try:
                module = importlib.import_module(module_path)
                skill_cls = getattr(module, class_name)
                skill_instance = skill_cls(cfg.get("config", {}))
                skill_instance.name = skill_name
                self.register(skill_instance)
                count += 1
            except (ImportError, AttributeError, Exception) as e:
                logger.error(f"[SkillRegistry] 加载技能 '{skill_name}' 失败: {e}")

        user_count = self._load_user_skills()

        total = count + user_count
        logger.info(f"[SkillRegistry] 技能加载完成: {total} 个 (系统{count} + 用户{user_count})")
        return total
