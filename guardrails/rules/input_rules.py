"""
输入校验规则 — 在用户输入进入 LangGraph 之前执行。
"""

from .base_rule import BaseRule, RuleResult, GuardrailsContext


class MaxLengthInputRule(BaseRule):
    """输入长度限制规则。"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "max_length_input"
        self.max_chars = config.get("max_chars", 4000)

    def check(self, ctx: GuardrailsContext) -> RuleResult:
        text = ctx.user_input
        if len(text) <= self.max_chars:
            return RuleResult()
        if self.action == "truncate":
            return RuleResult(
                passed=True,
                action="modify",
                message=config_msg(self),
                modified_content=text[:self.max_chars],
            )
        return RuleResult(passed=False, action="block", message=config_msg(self))


class BlockedTopicsRule(BaseRule):
    """敏感主题拦截规则。"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "blocked_topics"
        self.keywords = [kw.lower() for kw in config.get("keywords", [])]

    def check(self, ctx: GuardrailsContext) -> RuleResult:
        text_lower = ctx.user_input.lower()
        for kw in self.keywords:
            if kw in text_lower:
                return RuleResult(passed=False, action=self.action, message=config_msg(self))
        return RuleResult()


class PromptInjectionRule(BaseRule):
    """提示注入检测规则。"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "prompt_injection"
        self.patterns = [p.lower() for p in config.get("patterns", [])]

    def check(self, ctx: GuardrailsContext) -> RuleResult:
        text_lower = ctx.user_input.lower()
        for pat in self.patterns:
            if pat in text_lower:
                return RuleResult(passed=False, action=self.action, message=config_msg(self))
        return RuleResult()


def config_msg(rule: BaseRule) -> str:
    return rule.config.get("message", "")
