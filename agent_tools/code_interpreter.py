"""
代码执行沙箱模块 — 为 Agent 提供安全的 Python 代码运行环境。

安全机制：
  1. AST 语法检查 — 执行前预检查，避免语法错误导致运行时中断
  2. 安全导入拦截 — 基于 AST 分析，阻止 os/subprocess/socket 等危险模块
  3. 关键字黑名单 — 拦截 eval()/exec()/open() 等危险函数调用
  4. subprocess 隔离 — 在独立子进程中执行，空环境变量，防止污染宿主
  5. 30 秒超时 — 防止恶意死循环

对外只暴露一个 LangChain @tool：code_execute
"""

import ast
import subprocess
import sys
import tempfile
import os
import logging
from pathlib import Path
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# 沙箱目录（跨请求复用，避免重复创建临时目录）
_SANDBOX_DIR = None

# 禁止导入的模块列表 — 系统操作类模块，防止沙箱逃逸
_BLOCKED_IMPORTS = {
    "os", "subprocess", "sys", "shutil", "socket",
    "ctypes", "signal", "multiprocessing", "threading",
    "importlib", "builtins.__import__", "eval", "exec",
    "pickle", "shelve", "marshal",
}

# 禁止出现在源码中的函数调用字符串
_BLOCKED_KEYWORDS = {
    "__import__", "eval(", "exec(", "open(",
    "compile(", "getattr(", "setattr(", "delattr(",
}


def _syntax_check(code: str) -> str | None:
    """
    AST 语法预检查：在真实执行前捕获语法错误。
    Args:
        code: 用户提交的 Python 代码
    Returns: 有语法错误时返回错误描述字符串，否则返回 None
    """
    try:
        ast.parse(code)
    except SyntaxError as e:
        return f"Syntax error at line {e.lineno}: {e.msg}"
    return None


def _security_check(code: str) -> str | None:
    """
    安全检查：遍历 AST 节点树，阻止危险导入和函数调用。
    使用两层检测：
      1. 字符串匹配黑名单关键字（快速过滤）
      2. AST 精确分析 import 语句（防止字符串混淆绕过）
    Args:
        code: 用户提交的 Python 代码
    Returns: 拦截到时返回错误描述，否则返回 None
    """
    # 第一层：字符串级快速过滤
    for blocked in _BLOCKED_KEYWORDS:
        if blocked in code:
            return f"Security block: '{blocked}' is not allowed in code execution."
    # 第二层：AST 级精确分析 import 语句
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in _BLOCKED_IMPORTS:
                        return f"Security block: import '{alias.name}' is not allowed."
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in _BLOCKED_IMPORTS:
                    return f"Security block: import from '{node.module}' is not allowed."
    except SyntaxError:
        pass  # 语法检查在之前已完成，这里不会触发
    return None


def _get_sandbox_dir() -> str:
    """
    获取沙箱临时目录（全局复用，延迟创建）。
    Returns: 沙箱目录的绝对路径
    """
    global _SANDBOX_DIR
    if _SANDBOX_DIR is None:
        _SANDBOX_DIR = tempfile.mkdtemp(prefix="code_sandbox_")
    return _SANDBOX_DIR


@tool(description="执行 Python 代码并返回运行结果。支持 print 输出和返回值。代码在沙箱环境中运行，有 30 秒超时限制。")
def code_execute(code: str, timeout: int = 30) -> str:
    """
    执行用户提供的 Python 代码并返回运行结果。
    完整流程：语法检查 → 安全检查 → 写入临时文件 → subprocess 执行 → 清理。
    Args:
        code: 要执行的 Python 代码
        timeout: 超时秒数，默认 30 秒
    Returns: 标准输出/错误信息或执行失败原因
    """
    # 第一步：语法预检
    error = _syntax_check(code)
    if error:
        return f"[Code Executor] {error}"

    # 第二步：安全预检
    error = _security_check(code)
    if error:
        return f"[Code Executor] {error}"

    # 第三步：写入沙箱并执行
    sandbox = _get_sandbox_dir()
    script_path = os.path.join(sandbox, "_temp_script.py")
    try:
        # 用函数包裹用户代码 + try/except 确保异常被捕获
        with open(script_path, "w", encoding="utf-8") as f:
            f.write("import sys\n")
            f.write("def __result_wrapper():\n")
            for line in code.split("\n"):
                f.write(f"    {line}\n")
            f.write("\n")
            f.write("try:\n")
            f.write("    __result_wrapper()\n")
            f.write("except Exception as e:\n")
            f.write("    import traceback\n")
            f.write("    traceback.print_exc()\n")

        # 在子进程中执行（空环境变量，隔离系统环境）
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=sandbox,
            env={},  # 空环境变量，防止环境泄露
        )

        # 拼接 stdout 和 stderr 输出
        output_parts = []
        if result.stdout.strip():
            output_parts.append(f"[stdout]\n{result.stdout.strip()}")
        if result.stderr.strip():
            output_parts.append(f"[stderr]\n{result.stderr.strip()}")

        if not output_parts:
            return "[Code Executor] Code executed successfully (no output)."
        return "[Code Executor]\n" + "\n\n".join(output_parts)

    except subprocess.TimeoutExpired:
        return f"[Code Executor] Execution timed out after {timeout}s."
    except Exception as e:
        return f"[Code Executor] Execution error: {e}"
    finally:
        # 清理临时脚本文件
        if os.path.exists(script_path):
            os.remove(script_path)


# 导出为统一的工具变量名
code_exec_tool = code_execute
