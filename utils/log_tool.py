import os
import logging
from .path_tool import get_abs_path
from datetime import datetime

# 日志根目录
log_root = get_abs_path('logs')
os.makedirs(log_root, exist_ok = True)

# 日志默认格式
default_log_format = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(name)s - %(message)s'
)

def get_logger(
        name: str = 'agent',
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        log_file = None
) -> logging.Logger:
    """
    获取/创建日志管理器

    Args:
        name (str, optional): 名称. Defaults to 'agent'.
        console_level (int, optional): 控制台级别信息. Defaults to logging.INFO.
        file_level (int, optional): 文件级别信息. Defaults to logging.DEBUG.
        log_file (_type_, optional): 日志文件. Defaults to None.

    Returns:
        logging.Logger: 返回日志管理器对象
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 避免重复执行添加
    if logger.handlers:
        return logger
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(default_log_format)
    logger.addHandler(console_handler)

    if not log_file:
        log_file = os.path.join(log_root, f"{name}{datetime.now().strftime('%Y%m%d')}.log")

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(default_log_format)
    logger.addHandler(file_handler)

    return logger

logger = get_logger()