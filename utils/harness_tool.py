"""
Harness 工具 — 读取 AGENTS.md 约束规则并注入到 Agent System Prompt。

所有 Agent 通过 prompts_tool.py 加载提示词时自动追加规则，
确保 AGENTS.md 中的约束在运行时被每个 Agent 遵守。
"""

from .path_tool import get_abs_path
from .log_tool import logger

_HARNESS_CACHE: str | None = None


def get_harness_rules() -> str:
    """
    读取 AGENTS.md 中的约束规则。
    结果会缓存，避免重复读取文件。

    Returns:
        格式化的规则文本，如果文件不存在则返回空字符串
    """
    global _HARNESS_CACHE
    if _HARNESS_CACHE is not None:
        return _HARNESS_CACHE

    try:
        path = get_abs_path("AGENTS.md")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # 提取 # 号开头的章节和列表内容（去掉 Markdown 格式中的 --- 分隔线等）
        lines = []
        for line in content.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("---"):
                lines.append(stripped)
        rules = "\n".join(lines)
        _HARNESS_CACHE = "\n\n<Harness 约束规则>\n" + rules + "\n</Harness 约束规则>"
        logger.info(f"[Harness] 已加载 {len(lines)} 行约束规则")
        return _HARNESS_CACHE
    except FileNotFoundError:
        logger.warning("[Harness] AGENTS.md 未找到，跳过约束注入")
        _HARNESS_CACHE = ""
        return _HARNESS_CACHE
    except Exception as e:
        logger.error(f"[Harness] 加载 AGENTS.md 失败: {e}")
        _HARNESS_CACHE = ""
        return _HARNESS_CACHE
