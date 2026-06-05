from abc import ABC, abstractmethod
from typing import Any


class SkillResult:
    """技能执行结果的统一封装。"""

    def __init__(
        self,
        success: bool,
        output: str = "",
        error: str = "",
        data: dict[str, Any] | None = None,
    ):
        self.success = success
        self.output = output
        self.error = error
        self.data = data or {}

    def __str__(self) -> str:
        if self.success:
            return self.output
        return f"[Skill Error] {self.error}"


class BaseSkill(ABC):
    """技能抽象基类 — 所有技能的统一接口规范。

    子类必须实现：
      - name, description, category, keywords 类属性
      - execute() 方法

    可选覆写：
      - validate()  — 执行前参数校验
      - get_metadata() — 返回完整元数据字典
    """

    # ── 元数据（子类必须定义） ──────────────────────────
    name: str = ""               # 技能名称（唯一标识）
    description: str = ""        # 技能描述（LLM 理解用）
    category: str = "general"    # 分类标签
    keywords: list[str] = []     # 触发关键词（用于自动匹配）
    icon: str = "🔧"             # 图标（仅展示用）

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._enabled = self.config.get("enabled", True)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    def get_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "keywords": self.keywords,
            "enabled": self._enabled,
            "icon": self.icon,
        }

    def validate(self, **kwargs) -> str | None:
        """参数校验。返回 None 表示通过，返回字符串表示错误信息。"""
        return None

    @abstractmethod
    def execute(self, **kwargs) -> SkillResult:
        pass
