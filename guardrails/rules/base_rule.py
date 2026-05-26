"""
规则基类 — 所有 Guardrails 规则的抽象接口。

规则生命周期：
  1. 从 config/guardrails.yml 加载配置和规则列表
  2. 每条规则独立执行 check()
  3. 根据 RuleResult.action 决定如何处理：
     - allow:   放行
     - block:   拦截（返回错误）
     - warn:    仅记录警告，仍放行
     - modify:  修改内容后放行
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class RuleResult:
    """规则的检查结果。"""
    passed: bool = True
    action: str = "allow"
    message: str = ""
    modified_content: str = ""


@dataclass
class GuardrailsContext:
    """贯穿 Guardrails 全部检查阶段的上下文。"""
    user_input: str = ""
    route: str = ""
    session_id: str = ""
    final_output: str = ""
    violations: list = field(default_factory=list)


class BaseRule(ABC):
    """所有规则的抽象基类。"""

    def __init__(self, config: dict):
        self.name: str = ""
        self.config: dict = config
        self.enabled: bool = config.get("enabled", True)
        self.action: str = config.get("action", "block")

    @abstractmethod
    def check(self, ctx: GuardrailsContext) -> RuleResult:
        ...
