"""文件分析技能 — 读取、统计、分析项目文件内容。"""

import os
from pathlib import Path
from skills.base_skill import BaseSkill, SkillResult


class FileAnalysisSkill(BaseSkill):
    """文件分析技能。

    功能：
      - 统计项目文件数量、代码行数
      - 按文件类型分组统计
      - 分析代码复杂度
      - 检测重复文件

    用法示例：
      skill = FileAnalysisSkill()
      result = skill.execute(input="统计项目中的Python文件")
    """

    name = "file_analysis"
    description = "文件分析：统计代码行数、文件数量、项目结构分析"
    category = "development"
    keywords = ["文件", "统计", "分析", "代码", "行数", "项目", "结构", "目录", "大小"]
    icon = "📁"

    def execute(self, **kwargs) -> SkillResult:
        input_text = kwargs.get("input", "")
        max_size_kb = self.config.get("max_file_size_kb", 500)

        from utils.path_tool import get_project_root
        root = get_project_root()

        # 尝试从输入中提取目录路径，否则使用项目根目录
        target_dir = root
        words = input_text.split()
        for w in words:
            p = Path(w)
            if p.is_dir():
                target_dir = str(p.resolve())
                break

        stats = self._analyze_directory(target_dir, max_size_kb)

        output = (
            f"[FileAnalysis] 文件分析报告\n"
            f"目录: {target_dir}\n\n"
            f"[文件统计]\n"
            f"  总文件数: {stats['total_files']}\n"
            f"  总代码行数: {stats['total_lines']}\n"
            f"  总大小: {self._format_size(stats['total_size_bytes'])}\n\n"
            f"[按类型分布]\n"
        )

        for ext, info in sorted(stats["by_extension"].items()):
            output += f"  {ext}: {info['count']} 个文件, {info['lines']} 行, {self._format_size(info['size_bytes'])}\n"

        if stats.get("large_files"):
            output += f"\n[大文件（> 100KB）]\n"
            for f in stats["large_files"]:
                output += f"  {f['path']}: {self._format_size(f['size'])}\n"

        if stats.get("python_stats"):
            py = stats["python_stats"]
            output += f"\n[Python 统计]\n"
            output += f"  文件数: {py['file_count']}\n"
            output += f"  代码行: {py['code_lines']}\n"
            output += f"  注释行: {py['comment_lines']}\n"
            output += f"  空行: {py['blank_lines']}\n"

        return SkillResult(
            success=True,
            output=output,
            data=stats,
        )

    def _analyze_directory(self, root: str, max_size_kb: int) -> dict:
        """递归分析目录下的所有文件。"""
        stats = {
            "total_files": 0,
            "total_lines": 0,
            "total_size_bytes": 0,
            "by_extension": {},
            "large_files": [],
            "python_stats": {"file_count": 0, "code_lines": 0, "comment_lines": 0, "blank_lines": 0},
        }

        ignore_dirs = {".git", "__pycache__", ".venv", "node_modules", ".idea", "chroma_db"}
        ignore_exts = {".pyc", ".pyo", ".exe", ".dll", ".so", ".dylib", ".bin"}

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in ignore_dirs]

            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(fpath)
                except OSError:
                    continue

                _, ext = os.path.splitext(fname)
                ext = ext.lower() if ext else "(无扩展名)"

                if ext in ignore_exts:
                    continue

                stats["total_files"] += 1
                stats["total_size_bytes"] += size

                if ext not in stats["by_extension"]:
                    stats["by_extension"][ext] = {"count": 0, "lines": 0, "size_bytes": 0}
                stats["by_extension"][ext]["count"] += 1
                stats["by_extension"][ext]["size_bytes"] += size

                # 大文件标记
                if size > 100 * 1024:
                    rel_path = os.path.relpath(fpath, root)
                    stats["large_files"].append({"path": rel_path, "size": size})

                # 行数统计（仅文本文件）
                if ext in {".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".txt", ".yml", ".yaml",
                           ".json", ".html", ".css", ".scss", ".vue", ".java", ".go", ".rs", ".c", ".h"}:
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            lines = f.readlines()
                            line_count = len(lines)
                    except Exception:
                        line_count = 0

                    stats["total_lines"] += line_count
                    stats["by_extension"][ext]["lines"] += line_count

                    # Python 代码详细统计
                    if ext == ".py" and line_count <= max_size_kb * 2:
                        stats["python_stats"]["file_count"] += 1
                        code = comment = blank = 0
                        for line in lines:
                            stripped = line.strip()
                            if not stripped:
                                blank += 1
                            elif stripped.startswith("#"):
                                comment += 1
                            else:
                                code += 1
                        stats["python_stats"]["code_lines"] += code
                        stats["python_stats"]["comment_lines"] += comment
                        stats["python_stats"]["blank_lines"] += blank

        return stats

    @staticmethod
    def _format_size(bytes_val: int) -> str:
        if bytes_val < 1024:
            return f"{bytes_val} B"
        elif bytes_val < 1024 * 1024:
            return f"{bytes_val / 1024:.1f} KB"
        else:
            return f"{bytes_val / (1024 * 1024):.1f} MB"
