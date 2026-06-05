"""
动态技能（DynamicSkill）— 用户通过界面自定义的技能，无需写 Python 类。

原理：用户提供名称、描述、触发词、System Prompt，系统在 execute() 时
调用 LLM + 可选的工具列表，生成回答。

用户只需要填表单，不需要写任何代码。
"""

from typing import Any
from skills.base_skill import BaseSkill, SkillResult


class DynamicSkill(BaseSkill):
    """动态技能 — 通过配置就能生成，无需写 Python 类。

    用法（在 yml 中定义）：
      my_translator:
        enabled: true
        type: dynamic
        system_prompt: "你是一个翻译专家。将用户输入翻译成{target_lang}，只返回翻译结果。"
        config:
          target_lang: "英文"
        tools: []  # 可选：绑定的工具名列表

    或在代码中直接创建：
      skill = DynamicSkill({
          "name": "my_skill",
          "description": "...",
          "keywords": ["..."],
          "system_prompt": "...",
          "tools": ["web_search_tool"],
      })
    """

    name = "dynamic_skill"
    description = "用户自定义技能"
    category = "custom"
    keywords = []
    icon = "⚡"

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        cfg = config or {}

        # 从 config 中读取用户自定义元数据
        if cfg.get("name"):
            self.name = cfg["name"]
        if cfg.get("description"):
            self.description = cfg["description"]
        if cfg.get("keywords"):
            self.keywords = cfg["keywords"]
        if cfg.get("category"):
            self.category = cfg["category"]

        self.system_prompt = cfg.get("system_prompt", "你是一个AI助手，请回答用户的问题。")
        self.tool_names = cfg.get("tools", [])  # 绑定的工具名列表

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta["type"] = "dynamic"
        meta["system_prompt"] = self.system_prompt[:100]
        meta["tools"] = self.tool_names
        return meta

    def execute(self, **kwargs) -> SkillResult:
        """执行动态技能：用 LLM + 用户自定义 System Prompt 回答问题。"""
        from models.my_model import chat_model
        from langchain_core.messages import HumanMessage, SystemMessage

        input_text = kwargs.get("input", "").strip()
        if not input_text:
            return SkillResult(False, error="输入不能为空")

        messages = [SystemMessage(content=self.system_prompt)]

        # 注入会话记忆（如果有）
        memory = kwargs.get("memory_context", "")
        if memory:
            messages.append(SystemMessage(content=f"历史对话：\n{memory}"))

        messages.append(HumanMessage(content=input_text))

        try:
            # 如果有绑定工具，走 Agent 模式
            if self.tool_names:
                return self._execute_with_tools(input_text, memory)

            # 纯 LLM 模式
            result = chat_model.invoke(messages)
            return SkillResult(
                success=True,
                output=result.content,
                data={"skill_name": self.name},
            )

        except Exception as e:
            return SkillResult(False, error=f"技能执行失败: {e}")

    def _execute_with_tools(self, input_text: str, memory: str = "") -> SkillResult:
        """带工具的 Agent 模式。"""
        from langchain.agents import create_agent
        from agent_tools.middleware import (
            monitor_tool, log_before_model, performance_middleware,
        )
        from models.my_model import chat_model

        tools = self._resolve_tools()
        agent = create_agent(
            model=chat_model,
            system_prompt=self.system_prompt,
            tools=tools,
            middleware=[monitor_tool, log_before_model, performance_middleware],
        )

        messages = []
        if memory:
            messages.append({"role": "system", "content": f"历史对话：\n{memory}"})
        messages.append({"role": "user", "content": input_text})

        result = agent.invoke({"messages": messages})
        answer = result["messages"][-1].content

        return SkillResult(
            success=True,
            output=answer,
            data={"skill_name": self.name, "tools_used": self.tool_names},
        )

    def _resolve_tools(self):
        """根据工具名列表解析出实际的 LangChain tool 对象。"""
        resolved = []
        tool_map = {
            "web_search_tool": ("agent_tools.web_search_tool", "web_search_tool"),
            "rag_summarize": ("agent_tools.rag_tool", "rag_summarize"),
            "code_execute": ("agent_tools.code_interpreter", "code_execute"),
            "mcp_read_file": ("agent_tools.mcp_tools", "mcp_read_file"),
            "mcp_list_directory": ("agent_tools.mcp_tools", "mcp_list_directory"),
            "mcp_search_files": ("agent_tools.mcp_tools", "mcp_search_files"),
            "mcp_write_file": ("agent_tools.mcp_tools", "mcp_write_file"),
            "mcp_git_status": ("agent_tools.mcp_tools", "mcp_git_status"),
            "mcp_fetch_url": ("agent_tools.mcp_tools", "mcp_fetch_url"),
        }
        for name in self.tool_names:
            if name in tool_map:
                mod_path, attr = tool_map[name]
                try:
                    import importlib
                    mod = importlib.import_module(mod_path)
                    resolved.append(getattr(mod, attr))
                except Exception:
                    pass
        return resolved


def create_dynamic_skill_from_form(
    name: str,
    description: str,
    keywords: list[str],
    system_prompt: str,
    tools: list[str] | None = None,
    category: str = "custom",
) -> DynamicSkill:
    """从表单数据创建一个 DynamicSkill 实例。"""
    config = {
        "name": name,
        "description": description,
        "keywords": keywords,
        "category": category,
        "system_prompt": system_prompt,
        "tools": tools or [],
        "enabled": True,
    }
    return DynamicSkill(config)
