"""
输出校验规则 — 在 Agent 返回结果给用户之前执行。
"""

from .base_rule import BaseRule, RuleResult, GuardrailsContext


class SystemPromptLeakageRule(BaseRule):
    """系统提示泄露检测规则。"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "system_prompt_leakage"
        self.keywords = [kw.lower() for kw in config.get("keywords", [])]

    def check(self, ctx: GuardrailsContext) -> RuleResult:
        output_lower = ctx.final_output.lower()
        for kw in self.keywords:
            if kw in output_lower:
                return RuleResult(passed=False, action=self.action, message=config_msg(self))
        return RuleResult()


class MaxLengthOutputRule(BaseRule):
    """输出长度限制规则。"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "max_length_output"
        self.max_chars = config.get("max_chars", 50000)

    def check(self, ctx: GuardrailsContext) -> RuleResult:
        if len(ctx.final_output) <= self.max_chars:
            return RuleResult()
        if self.action == "truncate":
            return RuleResult(
                passed=True,
                action="modify",
                message=config_msg(self),
                modified_content=ctx.final_output[:self.max_chars],
            )
        return RuleResult(passed=False, action="block", message=config_msg(self))


def config_msg(rule: BaseRule) -> str:
    return rule.config.get("message", "")
