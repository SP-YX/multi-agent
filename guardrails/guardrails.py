"""
Guardrails — 多智能体系统的规则约束层（护栏）。

位于用户界面与 LangGraph 编排层之间，对所有进出流量进行规则校验：
  1. pre_process()  → 输入校验（长度、敏感词、注入检测）
  2. post_process() → 输出校验（系统提示泄露、长度限制）

所有规则通过 config/guardrails.yml 配置，支持热插拔。
"""

import logging
from typing import Optional
from .rules import (
    BaseRule, RuleResult, GuardrailsContext,
    MaxLengthInputRule, BlockedTopicsRule, PromptInjectionRule,
    SystemPromptLeakageRule, MaxLengthOutputRule,
)
from graph.agent_graph import agent_graph

logger = logging.getLogger(__name__)

# 规则名称 → 规则类的映射表
_RULE_REGISTRY: dict[str, type[BaseRule]] = {
    "max_length": MaxLengthInputRule,
    "blocked_topics": BlockedTopicsRule,
    "prompt_injection": PromptInjectionRule,
}

_OUTPUT_RULE_REGISTRY: dict[str, type[BaseRule]] = {
    "block_system_leakage": SystemPromptLeakageRule,
    "max_length": MaxLengthOutputRule,
}


class Guardrails:
    """Agent 规则约束引擎（护栏）。

    用法：
      guardrails = Guardrails()
      ok, cleaned, violations = guardrails.pre_process(user_input, session_id)
      if not ok:
          return violations  # 输入被拦截

      result = agent_graph.invoke(...)

      ok, final_output, violations = guardrails.post_process(result.get("final_answer"))
    """

    def __init__(self):
        self.enabled = True
        self.input_rules: list[BaseRule] = []
        self.output_rules: list[BaseRule] = []
        self.config = {}
        self._load_config()

    def _load_config(self):
        """加载 config/guardrails.yml 并初始化所有规则。"""
        try:
            from utils.config_tool import get_abs_path
            import yaml
            path = get_abs_path("config/guardrails.yml")
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            self.config = raw.get("guardrails", {})
        except Exception as e:
            logger.warning(f"Guardrails 配置加载失败，使用默认值: {e}")
            self.config = {"enabled": False}

        self.enabled = self.config.get("enabled", True)
        if not self.enabled:
            return

        input_cfg = self.config.get("input_rules", {})
        for rule_name, rule_cls in _RULE_REGISTRY.items():
            cfg = input_cfg.get(rule_name, {})
            if cfg.get("enabled", False):
                self.input_rules.append(rule_cls(cfg))

        output_cfg = self.config.get("output_rules", {})
        for rule_name, rule_cls in _OUTPUT_RULE_REGISTRY.items():
            cfg = output_cfg.get(rule_name, {})
            if cfg.get("enabled", False):
                self.output_rules.append(rule_cls(cfg))

    def pre_process(self, user_input: str, session_id: str = "") -> tuple[bool, str, list[dict]]:
        """输入预处理：校验 → 拦截/修改 → 放行。

        Args:
            user_input: 用户原始输入
            session_id: 会话 ID

        Returns:
            (ok, cleaned_input, violations)
            ok=False 表示输入被拦截，cleaned_input 为空字符串
        """
        if not self.enabled:
            return True, user_input, []

        ctx = GuardrailsContext(user_input=user_input, session_id=session_id)
        violations = []

        for rule in self.input_rules:
            result = rule.check(ctx)
            if not result.passed:
                entry = {
                    "rule": rule.name,
                    "action": result.action,
                    "message": result.message,
                }
                violations.append(entry)
                logger.warning(f"[Guardrails] 输入规则触犯 | rule={rule.name} action={result.action}")

                if result.action == "block":
                    return False, "", violations

            if result.action == "modify":
                ctx.user_input = result.modified_content

        return True, ctx.user_input, violations

    def post_process(
        self, final_output: str, ctx: Optional[GuardrailsContext] = None
    ) -> tuple[bool, str, list[dict]]:
        """输出后处理：校验 Agent 输出。

        Args:
            final_output: Agent 的最终输出
            ctx: 可选的上下文

        Returns:
            (ok, cleaned_output, violations)
        """
        if not self.enabled:
            return True, final_output, []

        if ctx is None:
            ctx = GuardrailsContext()

        ctx.final_output = final_output
        violations = []

        for rule in self.output_rules:
            result = rule.check(ctx)
            if not result.passed:
                entry = {
                    "rule": rule.name,
                    "action": result.action,
                    "message": result.message,
                }
                violations.append(entry)
                logger.warning(f"[Guardrails] 输出规则触犯 | rule={rule.name} action={result.action}")

                if result.action == "block":
                    return False, "", violations

            if result.action == "modify":
                ctx.final_output = result.modified_content

        return True, ctx.final_output, violations

    def wrap_graph(self, state: dict) -> dict:
        """一键包装 LangGraph invoke：先校验输入，执行 graph，再校验输出。

        Args:
            state: 传入 agent_graph.invoke() 的 state dict

        Returns:
            原始 state dict + 额外的 violation 信息
        """
        query = state.get("query", "")
        session_id = state.get("session_id", "")

        ok, cleaned, in_violations = self.pre_process(query, session_id)
        if not ok:
            return {
                **state,
                "final_answer": f"[Guardrails Blocked] {in_violations[0]['message'] if in_violations else '输入被拦截'}",
                "_guardrails_violations": in_violations,
                "_guardrails_blocked": True,
            }

        state["query"] = cleaned

        result = agent_graph.invoke(state)

        final = result.get("final_answer", "")
        ctx = GuardrailsContext(
            user_input=cleaned,
            session_id=session_id,
            final_output=final,
        )

        ok_out, cleaned_out, out_violations = self.post_process(final, ctx)
        result["_guardrails_violations"] = in_violations + out_violations
        result["_guardrails_blocked"] = not ok_out

        if not ok_out:
            result["final_answer"] = f"[Guardrails Blocked] {out_violations[0]['message'] if out_violations else '输出被拦截'}"

        return result


guardrails = Guardrails()