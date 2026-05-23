import os

def get_project_root() -> str:
    """
    获取工程根目录

    Returns:
        str: 返回工程根目录
    """
    cur_filepath = os.path.abspath(__file__)
    cur_dir = os.path.dirname(cur_filepath)
    return os.path.dirname(cur_dir)

def get_abs_path(relative_path: str) ->str:
    """
    获取绝对路径

    Args:
        relative_path (str): 相对路径

    Returns:
        str: 返回绝对路径
    """
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)