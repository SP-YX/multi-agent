"""
PyQt5 桌面客户端 — 多智能体系统的图形界面。

功能：
  - 聊天式对话界面
  - 中间结果可折叠查看（规划 / RAG / 搜索 / 代码）
  - 会话管理（新建 / 重置）
  - 异步执行，界面不卡顿
  - 性能统计展示

启动方式：
  python ui/qt_app.py
"""

import sys
import uuid
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QGroupBox,
    QSplitter, QFrame, QScrollArea, QStatusBar, QMessageBox,
    QMenuBar, QAction, QTabWidget, QGridLayout, QSizePolicy,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QTextCursor, QIcon

from dotenv import load_dotenv
load_dotenv()

from graph.agent_graph import agent_graph
from agent_tools.middleware import get_perf_stats, get_token_usage


# ═══════════════════════════════════════════════
# 后台工作线程 — 避免阻塞 UI
# ═══════════════════════════════════════════════

class AgentWorker(QThread):
    """异步执行 Agent 任务的工作线程."""
    finished = pyqtSignal(dict)     # 执行完成，发射结果字典
    error = pyqtSignal(str)         # 执行出错，发射错误消息

    def __init__(self, query: str, session_id: str):
        super().__init__()
        self.query = query
        self.session_id = session_id

    def run(self):
        try:
            result = agent_graph.invoke({
                "query": self.query,
                "session_id": self.session_id,
            })
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")


# ═══════════════════════════════════════════════
# 中间结果折叠面板
# ═══════════════════════════════════════════════

class CollapsiblePanel(QWidget):
    """可折叠展开的面板组件，用于显示中间结果。"""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setup_ui(title)

    def setup_ui(self, title: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # 折叠/展开切换按钮
        self.toggle_btn = QPushButton(f"▶  {title}")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(False)
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                text-align: left; padding: 6px 12px;
                background: #f0f0f0; border: 1px solid #ddd;
                border-radius: 4px; font-weight: bold;
            }
            QPushButton:checked { background: #e3f2fd; }
        """)
        self.toggle_btn.clicked.connect(self.on_toggle)
        layout.addWidget(self.toggle_btn)

        # 内容区域
        self.content = QTextEdit()
        self.content.setReadOnly(True)
        self.content.setMaximumHeight(200)
        self.content.setStyleSheet("""
            QTextEdit {
                background: #fafafa; border: 1px solid #ddd;
                border-top: none; border-radius: 0 0 4px 4px;
                padding: 8px; font-family: Consolas, monospace;
                font-size: 12px;
            }
        """)
        self.content.hide()
        layout.addWidget(self.content)

    def on_toggle(self, checked):
        """点击切换按钮时展开/折叠内容。"""
        self.content.setVisible(checked)
        self.toggle_btn.setText(f"▼  {self.toggle_btn.text()[2:]}" if checked
                                else f"▶  {self.toggle_btn.text()[2:]}")

    def set_content(self, text: str):
        """设置面板内容。"""
        self.content.setPlainText(text if text else "[No output]")


# ═══════════════════════════════════════════════
# 消息气泡自定义组件
# ═══════════════════════════════════════════════

class MessageBubble(QFrame):
    """单个消息气泡，左对齐（用户）或右对齐（助手）。"""

    def __init__(self, role: str, content: str, intermediate: dict = None, parent=None):
        super().__init__(parent)
        self.role = role
        self.intermediate = intermediate or {}
        self.setup_ui(content)

    def setup_ui(self, content: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        # 角色标签
        role_label = QLabel("🧑  You" if self.role == "user" else "🤖  Assistant")
        role_label.setStyleSheet("font-weight: bold; color: #555; font-size: 13px;")
        layout.addWidget(role_label)

        # 消息正文
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(content)
        text.setStyleSheet(f"""
            QTextEdit {{
                background: {"#e3f2fd" if self.role == "user" else "#f5f5f5"};
                border: none; border-radius: 8px; padding: 10px;
                font-size: 13px; color: #333;
            }}
        """)
        text.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        text.document().setDocumentMargin(0)
        # 自适应高度
        doc_height = text.document().size().height()
        text.setFixedHeight(int(doc_height) + 20)
        layout.addWidget(text)

        # 中间结果折叠面板
        if self.role == "assistant" and self.intermediate:
            for key, value in self.intermediate.items():
                label_map = {
                    "sub_tasks": "📋 任务规划",
                    "rag_result": "📚 RAG 检索",
                    "search_result": "🌐 联网搜索",
                    "code_result": "💻 代码执行",
                }
                panel = CollapsiblePanel(label_map.get(key, key))
                panel.set_content(value)
                layout.addWidget(panel)


# ═══════════════════════════════════════════════
# 主窗口
# ═══════════════════════════════════════════════

class MainWindow(QMainWindow):
    """多智能体系统主窗口。"""

    def __init__(self):
        super().__init__()
        self.session_id = str(uuid.uuid4())[:8]
        self.messages: list[dict] = []   # 聊天历史
        self.setup_ui()
        self.setup_menu()

    def setup_menu(self):
        """创建菜单栏。"""
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

    def setup_ui(self):
        """构建主界面布局。"""
        self.setWindowTitle("多智能体协作系统 v2.0")
        self.setMinimumSize(900, 700)
        self.setStyleSheet("""
            QMainWindow { background: #ffffff; }
            QGroupBox { font-weight: bold; border: 1px solid #ddd; border-radius: 6px; margin-top: 10px; padding-top: 16px; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
        """)

        # 中央部件
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── 顶部状态条 ──
        header = QFrame()
        header.setStyleSheet("background: #1565c0; padding: 12px;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 8, 16, 8)

        title_label = QLabel("🤖 多智能体协作系统")
        title_label.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
        header_layout.addWidget(title_label)

        self.session_label = QLabel(f"Session: {self.session_id}")
        self.session_label.setStyleSheet("color: rgba(255,255,255,0.8); font-size: 12px;")
        header_layout.addWidget(self.session_label)

        header_layout.addStretch()

        new_btn = QPushButton("🔄 新建会话")
        new_btn.setStyleSheet("""
            QPushButton { background: rgba(255,255,255,0.2); color: white; border: 1px solid rgba(255,255,255,0.4);
                          border-radius: 4px; padding: 6px 16px; font-size: 12px; }
            QPushButton:hover { background: rgba(255,255,255,0.3); }
        """)
        new_btn.clicked.connect(self.new_session)
        header_layout.addWidget(new_btn)
        main_layout.addWidget(header)

        # ── 消息显示区域 ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: #fafafa; }")
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.message_container = QWidget()
        self.message_layout = QVBoxLayout(self.message_container)
        self.message_layout.setAlignment(Qt.AlignTop)
        self.message_layout.setSpacing(8)
        self.message_layout.setContentsMargins(16, 16, 16, 16)
        # 占位拉伸
        self.message_layout.addStretch()

        scroll.setWidget(self.message_container)
        main_layout.addWidget(scroll, 1)

        # ── 底部输入区域 ──
        input_frame = QFrame()
        input_frame.setStyleSheet("background: #fff; border-top: 1px solid #ddd; padding: 12px;")
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(16, 8, 16, 8)

        self.input_edit = QTextEdit()
        self.input_edit.setPlaceholderText("输入你的任务...")
        self.input_edit.setMaximumHeight(80)
        self.input_edit.setStyleSheet("""
            QTextEdit { border: 1px solid #ccc; border-radius: 6px; padding: 8px;
                        font-size: 14px; background: #fafafa; }
            QTextEdit:focus { border-color: #1565c0; background: #fff; }
        """)
        input_layout.addWidget(self.input_edit, 1)

        self.send_btn = QPushButton("发送")
        self.send_btn.setMinimumSize(80, 40)
        self.send_btn.setStyleSheet("""
            QPushButton { background: #1565c0; color: white; border: none; border-radius: 6px;
                          font-size: 14px; font-weight: bold; }
            QPushButton:hover { background: #1976d2; }
            QPushButton:disabled { background: #bbb; }
        """)
        self.send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_btn)

        main_layout.addWidget(input_frame)

        # 状态栏
        self.status_bar = QStatusBar()
        self.status_bar.showMessage("就绪")
        self.setStatusBar(self.status_bar)

        # 快捷键：Enter 发送
        self.input_edit.installEventFilter(self)

    def eventFilter(self, obj, event):
        """捕获键盘事件：Ctrl+Enter 换行，Enter 发送。"""
        from PyQt5.QtCore import QEvent
        from PyQt5.QtGui import QKeyEvent
        if obj == self.input_edit and event.type() == QEvent.KeyPress:
            key_event = event
            if key_event.key() == Qt.Key_Return and not (key_event.modifiers() & Qt.ControlModifier):
                self.send_message()
                return True
        return super().eventFilter(obj, event)

    # ═══════════════════════════════════════════════
    # 核心逻辑
    # ═══════════════════════════════════════════════

    def add_message(self, role: str, content: str, intermediate: dict = None):
        """添加一条消息到界面和聊天历史。"""
        # 移除底部的占位拉伸
        self.message_layout.removeItem(self.message_layout.itemAt(self.message_layout.count() - 1))

        bubble = MessageBubble(role, content, intermediate)
        self.message_layout.addWidget(bubble)

        # 重新添加占位拉伸
        self.message_layout.addStretch()

        # 记录到聊天历史
        msg = {"role": role, "content": content}
        if intermediate:
            msg["intermediate"] = intermediate
        self.messages.append(msg)

        # 自动滚动到底部
        self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        """滚动消息区域到底部。"""
        scroll = self.findChild(QScrollArea)
        if scroll:
            scroll.verticalScrollBar().setValue(
                scroll.verticalScrollBar().maximum()
            )

    def send_message(self):
        """发送用户输入并启动 Agent 执行。"""
        text = self.input_edit.toPlainText().strip()
        if not text:
            return

        # 显示用户消息
        self.add_message("user", text)
        self.input_edit.clear()

        # 禁用发送按钮
        self.send_btn.setEnabled(False)
        self.send_btn.setText("⏳ 处理中...")
        self.status_bar.showMessage("正在执行任务...")

        # 启动工作线程
        self.worker = AgentWorker(text, self.session_id)
        self.worker.finished.connect(self.on_task_done)
        self.worker.error.connect(self.on_task_error)
        self.worker.start()

    def on_task_done(self, result: dict):
        """任务执行完成回调。"""
        final_answer = result.get("final_answer", "无输出")
        intermediate = {
            "sub_tasks": result.get("sub_tasks", ""),
            "rag_result": result.get("rag_result", ""),
            "search_result": result.get("search_result", ""),
            "code_result": result.get("code_result", ""),
        }
        self.add_message("assistant", final_answer, intermediate)
        self._reset_input()
        self.status_bar.showMessage("就绪", 3000)

    def on_task_error(self, error_msg: str):
        """任务执行出错回调。"""
        self.add_message("assistant", f"[Error] {error_msg}")
        self._reset_input()
        self.status_bar.showMessage(f"错误: {error_msg}")

    def _reset_input(self):
        """恢复输入区域状态。"""
        self.send_btn.setEnabled(True)
        self.send_btn.setText("发送")
        self.input_edit.setFocus()

    def new_session(self):
        """新建会话：重置 session_id 并清空聊天记录。"""
        self.session_id = str(uuid.uuid4())[:8]
        self.session_label.setText(f"Session: {self.session_id}")
        self.messages.clear()

        # 清空界面消息
        while self.message_layout.count() > 0:
            item = self.message_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.message_layout.addStretch()

        self.status_bar.showMessage(f"新会话已创建: {self.session_id}", 3000)

    def show_stats(self):
        """显示性能统计对话框。"""
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
        lines.append(f"📝 Token 消耗预估: {token.get('total', 0)}")
        QMessageBox.information(self, "性能统计", "\n".join(lines))


# ═══════════════════════════════════════════════
# 程序入口
# ═══════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
