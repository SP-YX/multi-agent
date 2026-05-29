"""
PyQt5 桌面客户端 — 多智能体系统

功能：
  - 侧边栏会话列表（类似 DeepSeek/豆包，持久化保存）
  - 聊天式对话界面，中间结果可折叠查看
  - 异步执行，界面不卡顿
  - 每次对话显示耗时
  - 性能统计 + Token 消耗展示

启动方式：
  python ui/qt_app.py
"""

import os
import json
import sys
import uuid
import time as _time
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QGroupBox,
    QSplitter, QFrame, QScrollArea, QStatusBar, QMessageBox,
    QMenuBar, QAction, QListWidget, QListWidgetItem, QSizePolicy,
    QDialog, QDialogButtonBox, QAbstractItemView,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QTextCursor, QIcon, QColor

from dotenv import load_dotenv
load_dotenv()

from guardrails import Guardrails
from agent_tools.middleware import get_perf_stats, get_token_usage

SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "memory", "sessions")


# 会话持久化辅助函数
def _list_sessions() -> list[dict]:
    if not os.path.isdir(SESSIONS_DIR):
        return []
    sessions = []
    for f in os.listdir(SESSIONS_DIR):
        if not f.endswith(".json"):
            continue
        path = os.path.join(SESSIONS_DIR, f)
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        sid = data.get("session_id", f.replace(".json", ""))
        window = data.get("window", [])
        title = window[0]["user"][:30] if window else "(空会话)"
        latest_ts = window[-1]["timestamp"] if window else ""
        sessions.append({
            "session_id": sid, "title": title,
            "latest_ts": latest_ts, "message_count": len(window),
        })
    sessions.sort(key=lambda s: s["latest_ts"], reverse=True)
    return sessions


def _load_session_messages(session_id: str) -> list[dict]:
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return []
    messages = []
    for turn in data.get("window", []):
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"], "intermediate": {}})
    return messages


def _delete_session_file(session_id: str):
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(path):
        os.remove(path)


# 后台工作线程
class AgentWorker(QThread):
    """异步执行 Agent 任务，带进度和耗时。"""
    progress = pyqtSignal(str)       # 当前节点名称
    finished = pyqtSignal(dict, float)  # 结果 + 耗时
    error = pyqtSignal(str)

    def __init__(self, query: str, session_id: str):
        super().__init__()
        self.query = query
        self.session_id = session_id
        self.guard = Guardrails()

    def run(self):
        t0 = _time.perf_counter()
        try:
            ok, cleaned, violations = self.guard.pre_process(self.query, self.session_id)
            if not ok:
                self.finished.emit({
                    "final_answer": f"[Guardrails Blocked] {violations[0].get('message', '输入被拦截')}",
                    "_guardrails_blocked": True,
                }, _time.perf_counter() - t0)
                return

            from graph.agent_graph import agent_graph
            initial_state = {"query": cleaned, "session_id": self.session_id}
            final_answer = ""
            intermediate = {}

            for event in agent_graph.stream(initial_state, stream_mode="updates"):
                for node_name, update in event.items():
                    label_map = {
                        "router": "分析问题类型…",
                        "simple_reply": "快速回复…",
                        "planner": "规划任务…",
                        "retrieval": "检索知识库 + 搜索网络…",
                        "coder": "执行代码…",
                        "summary": "汇总结果…",
                    }
                    self.progress.emit(label_map.get(node_name, node_name))
                    intermediate.update(update)
                    if "final_answer" in update:
                        final_answer = update["final_answer"]

            ok_out, final_cleaned, _ = self.guard.post_process(final_answer)
            if not ok_out:
                self.finished.emit({
                    "final_answer": "[Guardrails Blocked] 输出被拦截",
                    "_guardrails_blocked": True,
                }, _time.perf_counter() - t0)
                return

            result = {
                "final_answer": final_cleaned,
                "sub_tasks": intermediate.get("sub_tasks", ""),
                "rag_result": intermediate.get("rag_result", ""),
                "search_result": intermediate.get("search_result", ""),
                "code_result": intermediate.get("code_result", ""),
            }
            self.finished.emit(result, _time.perf_counter() - t0)
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")


# 中间结果折叠面板
class CollapsiblePanel(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.init_ui(title)

    def init_ui(self, title: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.toggle_btn = QPushButton(f"▶  {title}")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(False)
        self.toggle_btn.setStyleSheet("""
            QPushButton { text-align: left; padding: 4px 10px; background: #f0f0f0;
                          border: 1px solid #ddd; border-radius: 4px; font-weight: bold; font-size: 12px; }
            QPushButton:checked { background: #e3f2fd; }
        """)
        self.toggle_btn.clicked.connect(self.on_toggle)
        layout.addWidget(self.toggle_btn)

        self.content = QTextEdit()
        self.content.setReadOnly(True)
        self.content.setMaximumHeight(150)
        self.content.setStyleSheet("""
            QTextEdit { background: #fafafa; border: 1px solid #ddd; border-top: none;
                        border-radius: 0 0 4px 4px; padding: 6px; font-family: Consolas, monospace; font-size: 11px; }
        """)
        self.content.hide()
        layout.addWidget(self.content)

    def on_toggle(self, checked):
        self.content.setVisible(checked)
        prefix = "▼" if checked else "▶"
        self.toggle_btn.setText(f"{prefix}  {self.toggle_btn.text()[2:]}")

    def set_content(self, text: str):
        self.content.setPlainText(text if text else "[No output]")


# 消息气泡
class MessageBubble(QFrame):
    def __init__(self, role: str, content: str, intermediate: dict = None, elapsed: float = None, parent=None):
        super().__init__(parent)
        self.role = role
        self.intermediate = intermediate or {}
        self.elapsed = elapsed
        self.init_ui(content)

    def init_ui(self, content: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(4)

        role_label = QLabel("🧑 You" if self.role == "user" else "🤖 Assistant")
        role_label.setStyleSheet("font-weight: bold; color: #555; font-size: 12px;")
        layout.addWidget(role_label)

        bg = "#e3f2fd" if self.role == "user" else "#f5f5f5"
        text = QLabel(content)
        text.setWordWrap(True)
        text.setStyleSheet(f"""
            QLabel {{ background: {bg}; border: none; border-radius: 8px; padding: 10px;
                       font-size: 13px; color: #333; }}
        """)
        text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(text)

        # 耗时标签
        if self.role == "assistant" and self.elapsed is not None:
            time_label = QLabel(f"⏱️ 耗时 {self.elapsed:.1f}s")
            time_label.setStyleSheet("color: #999; font-size: 11px; padding-left: 4px;")
            layout.addWidget(time_label)

        if self.role == "assistant" and self.intermediate:
            label_map = {
                "sub_tasks": "任务规划", "rag_result": "RAG 检索",
                "search_result": "联网搜索", "code_result": "代码执行",
            }
            for key, value in self.intermediate.items():
                if value:
                    panel = CollapsiblePanel(label_map.get(key, key))
                    panel.set_content(value)
                    layout.addWidget(panel)


# ═══════════════════════════════════════════════
# 主窗口
# ═══════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.session_id = str(uuid.uuid4())[:8]
        self.messages: list[dict] = []
        self.init_ui()
        self.setup_menu()
        self.refresh_session_list()

    def setup_menu(self):
        menubar = self.menuBar()
        session_menu = menubar.addMenu("会话(&S)")
        new_action = QAction("新建会话(&N)", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_session)
        session_menu.addAction(new_action)

        view_menu = menubar.addMenu("视图(&V)")
        stats_action = QAction("性能统计(&P)", self)
        stats_action.setShortcut("Ctrl+P")
        stats_action.triggered.connect(self.show_stats)
        view_menu.addAction(stats_action)

    def init_ui(self):
        self.setWindowTitle("多智能体协作系统 v2.0")
        self.setMinimumSize(1100, 700)
        self.setStyleSheet("QMainWindow { background: #ffffff; }")

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 水平分割
        splitter = QSplitter(Qt.Horizontal)

        # 侧边栏
        sidebar = QFrame()
        sidebar.setStyleSheet("background: #f5f5f5; border-right: 1px solid #ddd;")
        sidebar.setMinimumWidth(220)
        sidebar.setMaximumWidth(320)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(6)

        sidebar_title = QLabel("💬 会话记录")
        sidebar_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #333;")
        sidebar_layout.addWidget(sidebar_title)

        new_btn = QPushButton("＋ 新建会话")
        new_btn.setStyleSheet("""
            QPushButton { background: #4CAF50; color: white; border: none; border-radius: 4px;
                          padding: 8px; font-size: 13px; font-weight: bold; }
            QPushButton:hover { background: #388E3C; }
        """)
        new_btn.clicked.connect(self.new_session)
        sidebar_layout.addWidget(new_btn)

        self.session_list = QListWidget()
        self.session_list.setStyleSheet("""
            QListWidget { border: none; background: transparent; font-size: 12px; }
            QListWidget::item { padding: 8px; border-bottom: 1px solid #eee; border-radius: 4px; }
            QListWidget::item:hover { background: #e8e8e8; }
            QListWidget::item:selected { background: #d0d0d0; }
        """)
        self.session_list.itemClicked.connect(self.on_session_clicked)
        sidebar_layout.addWidget(self.session_list, 1)

        # 当前会话标签 + Token
        self.sid_label = QLabel(f"当前: {self.session_id}")
        self.sid_label.setStyleSheet("color: #888; font-size: 11px;")
        sidebar_layout.addWidget(self.sid_label)

        self.token_label = QLabel("Token: -")
        self.token_label.setStyleSheet("color: #888; font-size: 11px;")
        sidebar_layout.addWidget(self.token_label)

        splitter.addWidget(sidebar)

        # 右侧聊天区
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # 顶部 header
        header = QFrame()
        header.setStyleSheet("background: #1565c0; padding: 8px 16px;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 8, 16, 8)

        title_label = QLabel("多智能体协作系统")
        title_label.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: rgba(255,255,255,0.9); font-size: 12px;")
        header_layout.addWidget(self.status_label)

        right_layout.addWidget(header)

        # 消息显示区
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: #fafafa; }")
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.message_container = QWidget()
        self.message_layout = QVBoxLayout(self.message_container)
        self.message_layout.setAlignment(Qt.AlignTop)
        self.message_layout.setSpacing(6)
        self.message_layout.setContentsMargins(16, 12, 16, 12)
        self.message_layout.addStretch()

        scroll.setWidget(self.message_container)
        right_layout.addWidget(scroll, 1)

        # 底部输入
        input_frame = QFrame()
        input_frame.setStyleSheet("background: #fff; border-top: 1px solid #ddd; padding: 8px 16px;")
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(16, 8, 16, 8)

        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText("输入你的任务…")
        self.input_edit.setMaximumHeight(70)
        self.input_edit.setStyleSheet("""
            QTextEdit { border: 1px solid #ccc; border-radius: 6px; padding: 8px;
                        font-size: 14px; background: #fafafa; }
            QTextEdit:focus { border-color: #1565c0; background: #fff; }
        """)
        input_layout.addWidget(self.input_edit, 1)

        self.send_btn = QPushButton("发送")
        self.send_btn.setMinimumSize(80, 38)
        self.send_btn.setStyleSheet("""
            QPushButton { background: #1565c0; color: white; border: none; border-radius: 6px;
                          font-size: 14px; font-weight: bold; padding: 8px 20px; }
            QPushButton:hover { background: #1976d2; }
            QPushButton:disabled { background: #bbb; }
        """)
        self.send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_btn)

        right_layout.addWidget(input_frame)

        splitter.addWidget(right_panel)
        splitter.setSizes([260, 840])
        main_layout.addWidget(splitter, 1)

        # 状态栏
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("就绪")
        self.setStatusBar(self.status_bar)

        self.input_edit.installEventFilter(self)
        self._update_token_display()

    def eventFilter(self, obj, event):
        from PyQt5.QtCore import QEvent
        from PyQt5.QtGui import QKeyEvent
        if obj == self.input_edit and event.type() == QEvent.KeyPress:
            key_event = event
            if key_event.key() == Qt.Key_Return and not (key_event.modifiers() & Qt.ControlModifier):
                self.send_message()
                return True
        return super().eventFilter(obj, event)

    # 会话管理
    def refresh_session_list(self):
        self.session_list.clear()
        sessions = _list_sessions()
        for sess in sessions:
            sid = sess["session_id"]
            label = sess["title"]
            if sess["message_count"] > 1:
                label += f" ({sess['message_count']}条)"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, sid)
            if sid == self.session_id:
                item.setBackground(QColor("#e3f2fd"))
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self.session_list.addItem(item)

    def on_session_clicked(self, item):
        sid = item.data(Qt.UserRole)
        if sid == self.session_id:
            return
        # 确认切换
        self.session_id = sid
        self.sid_label.setText(f"当前: {self.session_id[:8]}...")
        self.messages.clear()
        self._clear_chat_area()
        for msg in _load_session_messages(sid):
            elapsed = msg.get("elapsed")
            self.messages.append(msg)
            bubble = MessageBubble(msg["role"], msg["content"],
                                   msg.get("intermediate"), elapsed)
            self.message_layout.insertWidget(self.message_layout.count() - 1, bubble)
        self.refresh_session_list()
        self.status_bar.showMessage(f"已切换到会话 {sid[:8]}", 3000)

    def new_session(self):
        self.session_id = str(uuid.uuid4())[:8]
        self.sid_label.setText(f"当前: {self.session_id[:8]}...")
        self.messages.clear()
        self._clear_chat_area()
        self.status_label.setText("就绪")
        self.refresh_session_list()
        self.status_bar.showMessage(f"新会话已创建: {self.session_id[:8]}", 3000)

    def delete_session(self, sid: str):
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除此会话 {sid[:8]} 吗？此操作不可恢复。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        _delete_session_file(sid)
        if sid == self.session_id:
            self.new_session()
        else:
            self.refresh_session_list()
        self.status_bar.showMessage(f"会话 {sid[:8]} 已删除", 3000)

    # 聊天逻辑
    def _clear_chat_area(self):
        while self.message_layout.count() > 0:
            item = self.message_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.message_layout.addStretch()

    def add_message(self, role: str, content: str, intermediate: dict = None, elapsed: float = None):
        self.message_layout.removeItem(self.message_layout.itemAt(self.message_layout.count() - 1))
        bubble = MessageBubble(role, content, intermediate, elapsed)
        self.message_layout.addWidget(bubble)
        self.message_layout.addStretch()
        msg = {"role": role, "content": content}
        if intermediate:
            msg["intermediate"] = intermediate
        if elapsed is not None:
            msg["elapsed"] = elapsed
        self.messages.append(msg)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        scroll = self.findChild(QScrollArea)
        if scroll:
            scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())

    def _update_token_display(self):
        token = get_token_usage()
        self.token_label.setText(f"Token: {token.get('total', 0)}")

    def send_message(self):
        text = self.input_edit.toPlainText().strip()
        if not text:
            return

        self.add_message("user", text)
        self.input_edit.clear()
        self.send_btn.setEnabled(False)
        self.send_btn.setText("⏳ 处理中…")
        self.status_label.setText("⏳ 处理中…")
        self.status_bar.showMessage("正在执行任务…")

        self.worker = AgentWorker(text, self.session_id)
        self.worker.progress.connect(lambda s: self.status_label.setText(f"⏳ {s}"))
        self.worker.finished.connect(self.on_task_done)
        self.worker.error.connect(self.on_task_error)
        self.worker.start()

    def on_task_done(self, result: dict, elapsed: float):
        final_answer = result.get("final_answer", "无输出")

        if result.get("_guardrails_blocked"):
            self.add_message("assistant", final_answer, elapsed=elapsed)
            self._reset_input()
            self.status_bar.showMessage("Guardrails 拦截", 3000)
            return

        intermediate = {
            "sub_tasks": result.get("sub_tasks", ""),
            "rag_result": result.get("rag_result", ""),
            "search_result": result.get("search_result", ""),
            "code_result": result.get("code_result", ""),
        }
        has_inter = any(v for v in intermediate.values())
        self.add_message("assistant", final_answer,
                         intermediate if has_inter else None, elapsed)
        self._reset_input()
        self.status_label.setText(f"就绪（耗时 {elapsed:.1f}s）")
        self.status_bar.showMessage(f"处理完成，耗时 {elapsed:.1f}s", 5000)
        self.refresh_session_list()
        self._update_token_display()

    def on_task_error(self, error_msg: str):
        self.add_message("assistant", f"[System Error] {error_msg}")
        self._reset_input()
        self.status_label.setText("❌ 出错")
        self.status_bar.showMessage(f"错误: {error_msg}")

    def _reset_input(self):
        self.send_btn.setEnabled(True)
        self.send_btn.setText("发送")
        self.input_edit.setFocus()

    # 右键菜单（删除会话）
    def contextMenuEvent(self, event):
        item = self.session_list.itemAt(self.session_list.mapFromGlobal(event.globalPos()))
        if item:
            sid = item.data(Qt.UserRole)
            self.delete_session(sid)

    # 性能统计
    def show_stats(self):
        perf = get_perf_stats()
        token = get_token_usage()
        lines = ["📊 性能统计\n"]
        if perf:
            for name, stats in perf.items():
                lines.append(f"  {name}:")
                lines.append(f"    调用次数: {stats['count']}")
                lines.append(f"    平均耗时: {stats['avg_ms']:.1f}ms")
                lines.append(f"    最大耗时: {stats['max_ms']:.1f}ms")
                lines.append("")
        else:
            lines.append("  暂无数据\n")
        lines.append(f"Token 消耗预估: {token.get('total', 0)}")
        QMessageBox.information(self, "性能统计", "\n".join(lines))


# 程序入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())
