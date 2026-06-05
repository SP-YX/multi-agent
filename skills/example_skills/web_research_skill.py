"""联网调研技能 — 搜索、整理、总结互联网信息。"""

from skills.base_skill import BaseSkill, SkillResult


class WebResearchSkill(BaseSkill):
    """联网调研技能。

    功能：
      - 使用 Agent 内置的搜索引擎搜索互联网
      - 对搜索结果进行结构化整理
      - 自动汇总关键信息

    用法示例：
      skill = WebResearchSkill({"max_results": 5})
      result = skill.execute(input="2024年AI发展趋势")
    """

    name = "web_research"
    description = "联网调研：搜索互联网信息并整理总结"
    category = "research"
    keywords = ["搜索", "调研", "查询", "查找", "最新", "新闻", "资讯", "趋势", "报告"]
    icon = "🌐"

    def execute(self, **kwargs) -> SkillResult:
        input_text = kwargs.get("input", "")
        max_results = self.config.get("max_results", 5)

        if not input_text:
            return SkillResult(False, error="搜索关键词不能为空")

        try:
            from agent_tools.web_search_tool import web_search_tool

            search_result = web_search_tool.invoke({
                "query": input_text,
                "max_results": max_results,
            })

            if not search_result or "[Error]" in str(search_result):
                return SkillResult(
                    True,
                    output=f"[WebResearch] 搜索完成，未找到相关结果。\n查询: {input_text}",
                    data={"query": input_text, "results_count": 0},
                )

            output = (
                f"[WebResearch] 联网调研完成\n"
                f"查询: {input_text}\n"
                f"结果数: {max_results}\n\n"
                f"{search_result}"
            )

            return SkillResult(
                success=True,
                output=output,
                data={"query": input_text, "results_count": max_results},
            )

        except ImportError:
            return SkillResult(False, error="搜索工具不可用（缺少 web_search_tool 模块）")
        except Exception as e:
            return SkillResult(False, error=f"搜索失败: {e}")
