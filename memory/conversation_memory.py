"""
记忆系统模块 — 多智能体协作系统的对话历史管理

提供三级记忆体系：
  1. SlidingWindowMemory  — 滑动窗口记忆，保留最近 N 轮对话
  2. SummaryMemory        — 摘要记忆，超出窗口后压缩为历史摘要
  3. ConversationMemory   — 组合记忆（窗口 + 摘要 + JSON 文件持久化）

按 session_id 隔离不同会话，支持冷启动恢复。
"""

import json
import os
from datetime import datetime
from typing import Optional
from pathlib import Path


class SlidingWindowMemory:
    """滑动窗口记忆：始终保留最近 N 轮对话，超出则丢弃最旧记录。"""

    def __init__(self, window_size: int = 6):
        """
        初始化滑动窗口记忆。
        Args:
            window_size: 保留的最大对话轮数，默认为 6 轮
        """
        self.window_size = window_size
        # 历史记录列表，每个元素为 {"user": str, "assistant": str, "timestamp": str}
        self.history: list[dict] = []

    def add_turn(self, user: str, assistant: str):
        """
        添加一轮对话。若超出窗口大小，自动移除最旧记录。
        Args:
            user: 用户输入
            assistant: 模型回复
        """
        self.history.append({
            "user": user,
            "assistant": assistant,
            "timestamp": datetime.now().isoformat(),
        })
        # 超出窗口时，移除最旧的一条（队列 FIFO 行为）
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def format(self) -> str:
        """
        将历史记录格式化为可供 LLM 读取的文本。
        Returns: 格式化的对话历史字符串，无记录时返回空字符串
        """
        if not self.history:
            return ""
        lines = ["## Conversation History:"]
        for i, turn in enumerate(self.history, 1):
            lines.append(f"[Round {i}]")
            lines.append(f"  User: {turn['user']}")
            lines.append(f"  Assistant: {turn['assistant']}")
        return "\n".join(lines)

    def clear(self):
        """清空所有历史记录。"""
        self.history.clear()


class SummaryMemory:
    """摘要记忆：将超出窗口的旧对话压缩为一段摘要，保留长期信息。"""

    def __init__(self, max_summary_turns: int = 20):
        """
        初始化摘要记忆。
        Args:
            max_summary_turns: 每多少轮触发一次摘要更新条件，默认 20 轮
        """
        self.max_summary_turns = max_summary_turns
        self.summary: str = ""       # 历史摘要文本
        self.turn_count: int = 0     # 累计对话轮数

    def update_summary(self, new_summary: str):
        """
        更新历史摘要（由外部 LLM 调用生成）。
        Args:
            new_summary: LLM 生成的新摘要文本
        """
        self.summary = new_summary

    def add_turn(self):
        """递增对话轮次计数器。"""
        self.turn_count += 1

    def should_summarize(self) -> bool:
        """
        判断是否应该进行摘要更新（轮数达到阈值）。
        Returns: 达到阈值返回 True
        """
        return self.turn_count > 0 and self.turn_count % self.max_summary_turns == 0

    def format(self) -> str:
        """
        格式化摘要文本供 LLM 读取。
        Returns: 摘要文本，无摘要时返回空字符串
        """
        if not self.summary:
            return ""
        return f"## Historical Summary:\n{self.summary}"

    def clear(self):
        """清空摘要和计数器。"""
        self.summary = ""
        self.turn_count = 0


class ConversationMemory:
    """组合记忆：滑动窗口 + 摘要 + JSON 持久化，对外提供统一接口。"""

    def __init__(
        self,
        session_id: str = "default",
        window_size: int = 6,
        max_summary_turns: int = 20,
        persist_dir: Optional[str] = None,
    ):
        """
        初始化组合记忆。
        Args:
            session_id: 会话唯一标识，用于隔离不同对话
            window_size: 滑动窗口保留轮数
            max_summary_turns: 摘要触发间隔
            persist_dir: 持久化目录，默认为 memory/sessions/
        """
        self.session_id = session_id
        # 组合两个记忆层：短期窗口 + 长期摘要
        self.window = SlidingWindowMemory(window_size=window_size)
        self.summary_memory = SummaryMemory(max_summary_turns=max_summary_turns)
        self.persist_dir = persist_dir or os.path.join(
            os.path.dirname(__file__), "sessions"
        )
        # 启动时自动恢复历史记录
        self._load()

    def add_conversation(self, user: str, assistant: str):
        """
        添加一轮完整对话，同时更新窗口和计数器，并持久化。
        Args:
            user: 用户输入
            assistant: 系统回答
        """
        self.window.add_turn(user, assistant)
        self.summary_memory.add_turn()
        self._save()

    def get_context(self) -> str:
        """
        获取当前全部记忆上下文（摘要 + 窗口记录）。
        Returns: 格式化的记忆文本，按摘要→窗口顺序拼接
        """
        parts = []
        summary_text = self.summary_memory.format()
        if summary_text:
            parts.append(summary_text)
        window_text = self.window.format()
        if window_text:
            parts.append(window_text)
        return "\n\n".join(parts)

    def set_summary(self, summary: str):
        """
        手动设置摘要（可由外部 LLM 生成后注入）。
        Args:
            summary: 摘要文本
        """
        self.summary_memory.update_summary(summary)
        self._save()

    def clear(self):
        """清空当前会话的所有记忆并同步到磁盘。"""
        self.window.clear()
        self.summary_memory.clear()
        self._save()

    def _persist_path(self) -> str:
        """生成持久化文件路径：{persist_dir}/{session_id}.json。"""
        return os.path.join(self.persist_dir, f"{self.session_id}.json")

    def _save(self):
        """将记忆序列化为 JSON 写入磁盘，按 session_id 独立存储。"""
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        data = {
            "session_id": self.session_id,
            "window": self.window.history,
            "summary": self.summary_memory.summary,
            "turn_count": self.summary_memory.turn_count,
        }
        with open(self._persist_path(), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self):
        """从磁盘恢复记忆数据，文件不存在则静默跳过。"""
        path = self._persist_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 恢复窗口历史
            self.window.history = data.get("window", [])
            # 恢复摘要信息
            self.summary_memory.summary = data.get("summary", "")
            self.summary_memory.turn_count = data.get("turn_count", 0)
        except (json.JSONDecodeError, KeyError) as e:
            # 文件损坏时降级：打印日志但不清除现有记忆
            import logging
            logging.warning(f"Failed to load memory for {self.session_id}: {e}")
